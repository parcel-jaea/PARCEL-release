#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""性能試験の集計 (tests/perf/gen_perf.py の結果)。

  python3 tests/perf/parse_perf.py <ルート> [--dir perf_results] [--csv out.csv]

<ルート>/<dir>/<plat>/<構成>_np<N>/ を読む。<ルート> は試験を実行したディレクトリ (パッケージのルート) か、
そこから結果を集めた場所。NX を 512 以外にしたときは --dir perf_results_nx<NX>。

読み方:
  時間     rank 0 が回ごとに書く result<R>.txt (無ければ標準出力 run<R>.txt) から、ソルバ全体の時間
           ("... time = ... [sec] (total solver wall time ...)"、Q を作らない pcagmres は "(Q = ...)" の最後のブロック) と内訳を拾い、3 回のうち総時間が
           最短の回を採る。内訳もその回の値。
  反復     "converged_reason = <n> (<名前>) , its = N" の最後のもの (inverse_power は外側の反復、ks_arnoldi は再始動の回数)。
           <n> > 0 が収束。status が OK でも収束していなければ「未収束」と表示する。
  1 反復   最速の回の総時間 / 反復数 [ms]。
  装置数   ジョブが書く dev.txt。
"""
import argparse
import glob
import os
import re
import sys

# 拾う項目。表示名 -> 正規表現
METRICS = [
    # pcagmres は "(total solver wall time, Q = auto: formed)" のように後ろに書き足す。
    #   Q を作らない版 (implicit) は "(Q = implicit: not formed, CholeskyQR)" で、内訳を "main <項目> time" と書く
    ("total",  r"^\s*\S+ time\s*=\s*([0-9.eE+-]+)\s*\[sec\]\s*\((?:total solver wall time[,)]|Q = )"),
    ("init",   r"^\s*init time\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("loop",   r"^\s*main (?:loop )?time\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("dot",    r"^\s*(?:loop|main) dot(?: time)?\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("axpy",   r"^\s*(?:loop|main) axpy(?: time)?\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("spmv",   r"^\s*(?:loop|main) SpMV(?: time)?\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("precon", r"^\s*(?:loop|main) precon(?: time)?\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
    ("qr",     r"^\s*(?:loop|main) QR(?: time)?\s*=\s*([0-9.eE+-]+)\s*\[sec\]"),
]
RES = [(k, re.compile(p, re.M)) for k, p in METRICS]
RE_CONV = re.compile(r"converged_reason = (-?\d+) \((\w+)\)\s*,\s*its = (\d+)\s*,\s*residual = ([0-9.eE+-]+)")
CDP = "cd0.05_"


def metrics_of(text):
    """1 回の実行の出力から時間・反復数・収束状態を拾う。

    ソルバ全体の時間ブロックのうち**最後のもの**から拾う。inverse_power は内側のソルバを反復ごとに呼び、
    そのたびに内側のソルバのブロックを出力してから、最後に inverse_power 全体のブロックを出す。
    最後のブロックに無い項目 (inverse_power の SpMV など) は拾わない。
    """
    out = {}
    cs = list(RE_CONV.finditer(text))
    if cs:
        m = cs[-1]
        out["code"] = int(m.group(1))
        out["reason"] = m.group(2)
        out["its"] = int(m.group(3))
        out["resid"] = float(m.group(4))
        if len(cs) > 1:   # inverse_power: 内部ソルバの反復の合計
            out["inner_its"] = sum(int(x.group(3)) for x in cs[:-1])
    starts = [m.start() for m in RES[0][1].finditer(text)]
    if starts:
        text = text[starts[-1]:]
    for k, r in RES:
        m = r.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


def read_case(cdir):
    """1 構成のディレクトリ -> dict (status, dev, runs, 最速の回の値, 3 回の総時間の最小/最大)"""
    status = open(os.path.join(cdir, "status.txt")).read().strip()
    rec = {"status": status}
    f = os.path.join(cdir, "dev.txt")
    if os.path.isfile(f):
        rec["dev"] = int(open(f).read().strip() or 0)
    runs = []
    for pat in ("result[0-9]*.txt", "run[0-9]*.txt"):
        for r in sorted(glob.glob(os.path.join(cdir, pat))):
            if os.path.getsize(r) == 0:
                continue
            mm = metrics_of(open(r, encoding="utf-8", errors="replace").read())
            if "total" in mm:
                runs.append(mm)
        if runs:
            break
    rec["runs"] = len(runs)
    if runs:
        best = min(runs, key=lambda x: x["total"])
        rec.update(best)
        rec["total_max"] = max(x["total"] for x in runs)
        its = {x.get("its") for x in runs}
        rec["its_same"] = len(its) == 1
        if best.get("its"):
            rec["ms_per_it"] = best["total"] / best["its"] * 1e3
    if "code" in rec and rec["code"] <= 0 and status == "OK":
        rec["status"] = "未収束:" + rec.get("reason", "?")
    return rec


def collect(root, sub="perf_results"):
    """{plat: {(label, np): rec}}"""
    data = {}
    base = os.path.join(root, sub)
    for plat in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        pdir = os.path.join(base, plat)
        if not os.path.isdir(pdir):
            continue
        out = {}
        for d in sorted(os.listdir(pdir)):
            cdir = os.path.join(pdir, d)
            m = re.match(r"(.+)_np(\d+)$", d)
            if not m or not os.path.isfile(os.path.join(cdir, "status.txt")):
                continue
            out[(m.group(1), int(m.group(2)))] = read_case(cdir)
        data[plat] = out
    return data


def matrix_of(label):
    return "convdiff" if label.startswith(CDP) else "poisson"


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="性能試験の集計")
    ap.add_argument("root")
    ap.add_argument("--dir", default="perf_results", help="結果のディレクトリ (既定 perf_results)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    data = collect(a.root, a.dir)
    if not data:
        print("結果が見つかりません"); sys.exit(1)

    rows = []
    for plat in sorted(data):
        d = data[plat]
        labels = sorted({l for l, _ in d})
        devs = sorted({r.get("dev", n) for (_, n), r in d.items()})
        print(f"\n######## {plat}  総時間 [s] / 反復数 / 1 反復 [ms] (各 3 回の最速) ########")
        print(f"  {'構成':<42}" + "".join(f"{'装置数 ' + str(v):>30}" for v in devs) + f"{'強スケーリング':>18}")
        for lab in labels:
            by_dev = {r.get("dev", n): (n, r) for (l, n), r in d.items() if l == lab}
            cells = []
            for v in devs:
                n, r = by_dev.get(v, (None, {}))
                if r.get("status") != "OK" or r.get("total") is None:
                    cells.append(f"{r.get('status', '-')[:28]:>30}")
                else:
                    mark = " ※" if r.get("total_max", 0) > r["total"] * 1.5 else ""
                    cells.append(f"{fmt(r['total'], 2) + ' / ' + str(r.get('its', '-')) + ' / ' + fmt(r.get('ms_per_it'), 2) + mark:>30}")
                rows.append(dict(plat=plat, label=lab, matrix=matrix_of(lab), dev=v, np=n, status=r.get("status"),
                                 its=r.get("its"), reason=r.get("reason"), ms_per_it=r.get("ms_per_it"),
                                 total_max=r.get("total_max"), runs=r.get("runs"),
                                 **{k: r.get(k) for k, _ in METRICS}))
            # 最小の装置数から最大の装置数への総時間の速度向上
            sc = "-"
            lo, hi = by_dev.get(devs[0], (None, {}))[1], by_dev.get(devs[-1], (None, {}))[1]
            if lo.get("status") == "OK" and hi.get("status") == "OK" and lo.get("total") and hi.get("total"):
                sc = f"x{lo['total'] / hi['total']:.2f} ({devs[0]}->{devs[-1]})"
            print(f"  {lab:<42}" + "".join(cells) + f"{sc:>18}")
    print("\n  ※ 3 回の最長が最短の 1.5 倍を超える構成。未収束: status は OK だが converged_reason <= 0")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  CSV: {a.csv} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
