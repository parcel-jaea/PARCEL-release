#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""MPI なしの回帰試験 (<plat>_nompi) を、同じ環境の MPI あり np=1 と突き合わせる。

  python3 tests/regression/compare_nompi.py [<plat> ...] [--root .] [--md out.md] [--show 30]
  python3 tests/regression/compare_nompi.py --csv iters_*.csv [<plat> ...]

MPI なし (-DPARCEL_MPI=0、mpirun なしの 1 プロセス) と MPI あり np=1 は同じ計算をするので、
状態・反復数・誤差は一致するはず。
合否の判定はそれぞれの回帰試験が済ませているので、ここでは一致しているかだけを見る。

既定 (結果のディレクトリを読む。両方の結果があるホストで、パッケージのルートから):
  regression_results/<plat>/<精度>/<ケース>_np1/ と regression_results/<plat>_nompi/<精度>/<ケース>_np1/
  (ddcheck/ddcheck_np1 を含む) を組にして、次を比べる。
    状態       status.txt
    反復数     反復履歴の最後の "N : residual norm" の N、外側の "iterations / restarts = N"
    数値       参照解との誤差 (線形: x_direct との相対誤差、固有値: max relative eigenvalue err)、
               真の残差 (finish : residual true norm)、固有値 (Eigen Value = / 表の Re・Im)、[DDCHECK] の行
    出力全体   時間の行・AdaptiveCpp の警告を除いた出力 (parcel_result.txt、無ければ result.txt)
  <plat> を省略すると regression_results/ にある *_nompi をすべて対象にする。

--csv (collect_iters.sh の CSV。結果のディレクトリが手元に無いとき):
  plat=<plat>_nompi と plat=<plat> の np=1 を突き合わせ、状態と反復数だけを比べる。

終了コード: 両方にある組で状態か反復数か数値が 1 件でも違えば 1、そうでなければ 0
(片方にしか無い組は数えるだけで、終了コードには効かない)。
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import OrderedDict, defaultdict

NOMPI = "_nompi"
ANSI = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")
# 実行ごとに変わる行 (時間・警告・探索パス)。これを除いた出力が一致するかを見る
VOLATILE = re.compile(r"\[sec\]|AdaptiveCpp|github\.com/AdaptiveCpp|秒|elapsed|wall ?time", re.I)
IT_RE = re.compile(r"^ *([0-9]+) : residual norm")
OUTER_RE = re.compile(r"(?:iterations|restarts) *= *([0-9]+)")
TRUE_RE = re.compile(r"finish : residual true norm.*=\s*(\S+)")
EIG_RE = re.compile(r"Eigen Value\s*=\s*(\S+)")
# ks_arnoldi の検証表の行: j Re Im Re_ana Im_ana eig_err residual
EIGROW_RE = re.compile(r"^\s*\d+\s+(\S+e[+-]\d+)\s+(\S+e[+-]\d+)\s+\S+e[+-]\d+\s+\S+e[+-]\d+\s+\S+\s+\S+\s*$")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def result_file(d, other):
    """判定に使った出力。両方に parcel_result.txt があればそれ、無ければ result.txt"""
    nonempty = lambda f: os.path.exists(f) and os.path.getsize(f) > 0
    for name in ("parcel_result.txt", "ddcheck_result.txt"):
        if nonempty(os.path.join(d, name)) and nonempty(os.path.join(other, name)):
            return os.path.join(d, name)
    return os.path.join(d, "result.txt")


def extract(text):
    """比べる値を取り出す"""
    v = OrderedDict()
    lines = ANSI.sub("", text or "").splitlines()
    its = [IT_RE.match(l).group(1) for l in lines if IT_RE.match(l)]
    v["iters"] = its[-1] if its else ""
    outs = [m.group(1) for l in lines for m in [OUTER_RE.search(l)] if m]
    v["outer"] = outs[-1] if outs else ""
    err = ""
    for l in lines:
        if "x_direct| / max" in l:
            err = l.split()[-1]; break
    if not err:
        for l in lines:
            if "max relative eigenvalue err" in l:
                f = l.split()
                err = f[5] if len(f) > 5 else ""; break
    v["err"] = err
    tr = [m.group(1) for l in lines for m in [TRUE_RE.search(l)] if m]
    v["true_res"] = tr[-1] if tr else ""
    eig = [m.group(1) for l in lines for m in [EIG_RE.search(l)] if m]
    eig += [m.group(1) + "," + m.group(2) for l in lines for m in [EIGROW_RE.match(l)] if m]
    v["eig"] = ";".join(eig)
    v["ddcheck"] = ";".join(l.strip() for l in lines if l.startswith("[DDCHECK]"))
    return v


def normalized(text):
    out = []
    for l in ANSI.sub("", text or "").splitlines():
        if VOLATILE.search(l):
            continue
        out.append(l.rstrip())
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"{i + 1} 行目: MPI あり `{x.strip()[:80]}` / なし `{y.strip()[:80]}`"
    return f"行数 {len(a)} / {len(b)}"


def cases_of(root, plat):
    """{(精度, ケース): ディレクトリ} (np=1 のものだけ)"""
    out = {}
    for f in glob.glob(os.path.join(root, "regression_results", plat, "*", "*_np1", "status.txt")):
        d = os.path.dirname(f)
        prec = os.path.basename(os.path.dirname(d))
        case = os.path.basename(d)[:-len("_np1")]
        out[(prec, case)] = d
    return out


def compare_dirs(root, base):
    a, b = cases_of(root, base), cases_of(root, base + NOMPI)
    rows = []
    for key in sorted(set(a) | set(b)):
        r = dict(plat=base, prec=key[0], case=key[1])
        if key not in a or key not in b:
            r["kind"] = "MPI ありのみ" if key in a else "MPI なしのみ"
            rows.append(r); continue
        da, db = a[key], b[key]
        sa = (read(os.path.join(da, "status.txt")) or "").strip()
        sb = (read(os.path.join(db, "status.txt")) or "").strip()
        ta = read(result_file(da, db)) or ""
        tb = read(result_file(db, da)) or ""
        va, vb = extract(ta), extract(tb)
        r.update(status_a=sa, status_b=sb, va=va, vb=vb)
        r["status_eq"] = sa == sb
        r["iter_eq"] = va["iters"] == vb["iters"] and va["outer"] == vb["outer"]
        r["num_eq"] = all(va[k] == vb[k] for k in ("err", "true_res", "eig", "ddcheck"))
        na, nb = normalized(ta), normalized(tb)
        r["text_eq"] = na == nb
        r["text_diff"] = "" if r["text_eq"] else first_diff(na, nb)
        r["kind"] = "両方"
        rows.append(r)
    return rows


def compare_csv(paths, bases):
    it = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["np"] != "1":
                continue
            it[(r["plat"], r["prec"], r["case"])] = r
    if not bases:
        bases = sorted({k[0][:-len(NOMPI)] for k in it if k[0].endswith(NOMPI)})
    rows = []
    for base in bases:
        keys = sorted({(k[1], k[2]) for k in it if k[0] in (base, base + NOMPI)})
        for prec, case in keys:
            ra, rb = it.get((base, prec, case)), it.get((base + NOMPI, prec, case))
            r = dict(plat=base, prec=prec, case=case)
            if ra is None or rb is None:
                r["kind"] = "MPI ありのみ" if ra else "MPI なしのみ"
                rows.append(r); continue
            va = dict(iters=ra["iters"], outer=ra["outer"])
            vb = dict(iters=rb["iters"], outer=rb["outer"])
            r.update(kind="両方", status_a=ra["status"], status_b=rb["status"], va=va, vb=vb,
                     status_eq=ra["status"] == rb["status"], iter_eq=va == vb,
                     num_eq=None, text_eq=None, text_diff="")
            rows.append(r)
    return rows, bases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plats", nargs="*", help="MPI ありの環境名 (s86_cpu など。_nompi は付けても付けなくてもよい)")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
                    help="regression_results/ のあるディレクトリ (既定: パッケージのルート)")
    ap.add_argument("--csv", nargs="+", help="collect_iters.sh の CSV (状態と反復数だけを比べる)")
    ap.add_argument("--md", help="結果を Markdown で書き出す")
    ap.add_argument("--show", type=int, default=50, help="一覧に出す不一致の最大件数")
    a = ap.parse_args()
    bases = [p[:-len(NOMPI)] if p.endswith(NOMPI) else p for p in a.plats]

    if a.csv:
        rows, bases = compare_csv(a.csv, bases)
        mode = "CSV (状態・反復数のみ): " + ", ".join(a.csv)
    else:
        if not bases:
            bases = sorted(os.path.basename(p)[:-len(NOMPI)] for p in
                           glob.glob(os.path.join(a.root, "regression_results", "*" + NOMPI)))
        rows = [r for b in bases for r in compare_dirs(a.root, b)]
        mode = "結果のディレクトリ: " + os.path.join(os.path.abspath(a.root), "regression_results")
    if not bases:
        sys.exit("対象の環境が無い (regression_results/<plat>_nompi も CSV の <plat>_nompi も無い)")

    out = ["# MPI なしと MPI あり np=1 の突き合わせ\n", f"入力: {mode}\n",
           "| 環境 | 両方にある組 | 状態一致 | 反復数一致 | 数値一致 | 出力全体一致 | MPI ありのみ | MPI なしのみ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    by = defaultdict(list)
    for r in rows:
        by[r["plat"]].append(r)
    bad = []
    for base in bases:
        rs = by.get(base, [])
        both = [r for r in rs if r["kind"] == "両方"]
        cnt = lambda k: "-" if both and both[0][k] is None else str(sum(bool(r[k]) for r in both))
        out.append(f"| {base} | {len(both)} | {cnt('status_eq') if both else 0} | {cnt('iter_eq') if both else 0} | "
                   f"{cnt('num_eq') if both else 0} | {cnt('text_eq') if both else 0} | "
                   f"{sum(r['kind'] == 'MPI ありのみ' for r in rs)} | {sum(r['kind'] == 'MPI なしのみ' for r in rs)} |")
        bad += [r for r in both if not (r["status_eq"] and r["iter_eq"] and r["num_eq"] is not False)]
    out.append(f"\n## 状態・反復数・数値が違う組 ({len(bad)} 件)\n")
    if bad:
        out.append("| 環境 | 精度 | ケース | 状態 (あり / なし) | 反復数 (あり / なし) | 誤差 (あり / なし) | 違う項目 |")
        out.append("|---|---|---|---|---|---|---|")
        for r in bad[:a.show]:
            va, vb = r["va"], r["vb"]
            what = [k for k in ("status_eq", "iter_eq", "num_eq") if r[k] is False]
            if r["num_eq"] is False:
                what += [k for k in ("err", "true_res", "eig", "ddcheck") if va.get(k) != vb.get(k)]
            out.append(f"| {r['plat']} | {r['prec']} | `{r['case']}` | {r['status_a']} / {r['status_b']} | "
                       f"{va['iters']}:{va['outer']} / {vb['iters']}:{vb['outer']} | "
                       f"{va.get('err', '')} / {vb.get('err', '')} | {', '.join(what)} |")
        if len(bad) > a.show:
            out.append(f"\n(ほか {len(bad) - a.show} 件。--show で増やせる)")
    if not a.csv:
        textdiff = [r for r in rows if r["kind"] == "両方" and not r["text_eq"]
                    and r["status_eq"] and r["iter_eq"] and r["num_eq"]]
        out.append(f"\n## 状態・反復数・数値は一致し、出力のほかの行だけが違う組 ({len(textdiff)} 件)\n")
        for r in textdiff[:a.show]:
            out.append(f"- {r['plat']} / {r['prec']} / `{r['case']}`: {r['text_diff']}")
    only = [r for r in rows if r["kind"] != "両方"]
    if only:
        out.append(f"\n## 片方にしか無い組 ({len(only)} 件、先頭 {min(len(only), a.show)} 件)\n")
        for r in only[:a.show]:
            out.append(f"- {r['plat']} / {r['prec']} / `{r['case']}`: {r['kind']}")
    text = "\n".join(out) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    print(f"組 {sum(r['kind'] == '両方' for r in rows)}、不一致 {len(bad)}、片方のみ {len(only)}",
          file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
