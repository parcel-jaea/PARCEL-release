#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""回帰試験の反復数を環境間で突き合わせる (合否は変えない、報告のみ)。

  python3 tests/regression/compare_iters.py iters_*.csv [--md out.md] [--csv out.csv]
          [--spread 0.2] [--vs-double 3.0]

入力は tests/regression/collect_iters.sh の出力 (1 ホスト 1 ファイル、何本でも)。

低精度の構成では、縮約の順序 (スレッド数・ランク数・装置で変わる) による倍精度の
小さな差が低精度への丸めで増幅され、反復数が環境で変わる。回帰の合否は誤差と収束しか見ないので、この違いは合格のまま
埋もれる。ここでは次の 2 つを数え、閾値を超えた組を「注意」として並べる。

  環境間の幅   同じ (精度, ケース, np) の反復数の max/min - 1   (--spread、既定 0.2 = 20%)
  double 比    同じ (ケース, np, 環境) の double に対する倍率    (--vs-double、既定 3.0)

反復数が小さい組 (ks_arnoldi のリスタート数 1〜3 など) は 1 回の差が 100% になるので、
幅の注意は反復数が --min-count (既定 10) 以上の組に限る (値は表に残す)。
精度は gen_regression.py の 13 種だけを数える (それ以外の名前の結果は読み飛ばす)。
ブロックヤコビ (p2) は前処理そのものが環境 (装置の並列度) と MPI 分割で変わるので、
環境間の幅は数えない (違って当然)。double 比は数える。
反復数は線形ソルバ・内部解法では反復履歴の最後の番号、固有値ソルバでは外側の反復数
(inverse_power) / リスタート数 (ks_arnoldi) を使う。
"""
import argparse
import csv
import sys
from collections import defaultdict

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from site_config import REGRESSION  # noqa: E402

# 表の列の順 (CPU 環境、GPU 環境の順) と CPU 環境の集合は tests/site_config.py から取る
CPU = {p for p, c in REGRESSION.items() if not c.get("gpu")}
PLAT_ORDER = [p for p in REGRESSION if p in CPU] + [p for p in REGRESSION if p not in CPU]
# MPI なしの構成 (<plat>_nompi) は元の環境の直後に並べ、CPU / GPU も元の環境に従う。
#   MPI あり np=1 との一致は compare_nompi.py で確かめる。
PLAT_ORDER = [q for p in PLAT_ORDER for q in (p, p + "_nompi")]
CPU = CPU | {p + "_nompi" for p in CPU}
PREC_ORDER = ["double", "calcDD", "dataDD", "precon_dataF", "precon_calcF", "precon_allF",
              "precon_dataF16", "precon_calcF16", "precon_allF16", "calcF", "dataF", "calcF16", "dataF16"]
OUTER_SOLVERS = ("power", "inverse_power", "invpow_pspec", "ks_arnoldi", "kssi_pspec", "ksqr_")


def count_of(r):
    base = r["case"]
    use_outer = base.startswith(OUTER_SOLVERS)
    v = r["outer"] if use_outer else r["iters"]
    if not use_outer and not v:
        v = r["outer"]
    return int(v) if v and v.isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--csv")
    ap.add_argument("--spread", type=float, default=0.2)
    ap.add_argument("--vs-double", type=float, default=3.0)
    ap.add_argument("--min-count", type=int, default=10)
    a = ap.parse_args()

    it = {}          # (prec, case, np, plat) -> 反復数 (OK のものだけ)
    date = defaultdict(set)
    for path in a.inputs:
        for r in csv.DictReader(open(path)):
            if r["status"] != "OK" or r["prec"] not in PREC_ORDER:
                continue
            c = count_of(r)
            if c is None:
                continue
            it[(r["prec"], r["case"], r["np"], r["plat"])] = c
            date[r["plat"]].add(r["date"])
    plats = [p for p in PLAT_ORDER if any(k[3] == p for k in it)] + \
            sorted({k[3] for k in it} - set(PLAT_ORDER))

    groups = defaultdict(dict)
    for (prec, case, np_, plat), c in it.items():
        groups[(prec, case, np_)][plat] = c

    rows = []
    for (prec, case, np_), m in groups.items():
        vals = list(m.values())
        spread = (max(vals) / min(vals) - 1.0) if len(vals) >= 2 and min(vals) > 0 else None
        vsd = {}
        for plat, c in m.items():
            d = it.get(("double", case, np_, plat))
            if d and prec != "double":
                vsd[plat] = c / d
        cpu = sorted({m[p] for p in m if p in CPU}); gpu = sorted({m[p] for p in m if p not in CPU})
        split = bool(cpu and gpu and not set(cpu) & set(gpu))
        rows.append(dict(prec=prec, case=case, np=np_, n=len(vals), spread=spread,
                         vs_double_max=max(vsd.values()) if vsd else None,
                         cpu_gpu_split=split, **{p: m.get(p, "") for p in plats}))

    porder = {p: i for i, p in enumerate(PREC_ORDER)}
    rows.sort(key=lambda r: (porder.get(r["prec"], 99), r["case"], int(r["np"])))

    def warn_spread(r):
        vals = [r[p] for p in plats if r[p] != ""]
        return (r["spread"] is not None and r["spread"] > a.spread and "_p2_" not in r["case"] + "_"
                and vals and max(vals) >= a.min_count)
    def warn_vsd(r):
        return r["vs_double_max"] is not None and r["vs_double_max"] > a.vs_double

    out = []
    out.append(f"# 回帰試験の反復数の環境間比較\n")
    out.append(f"環境と結果の日付: " + "、".join(f"{p}（{'/'.join(sorted(date[p]))}）" for p in plats) + "\n")
    out.append(f"閾値: 環境間の幅 > {a.spread:.0%}（ブロックヤコビと反復数 {a.min_count} 未満は除く）、double 比 > {a.vs_double:g} 倍。"
               f"**合否は変えない（報告のみ）。**\n")
    out.append("## 精度ごとの要約\n")
    out.append("| 精度 | 組 | 環境で反復数が違う組 | うち CPU と GPU で分かれる | 幅 > 閾値 | double 比 > 閾値 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    by = defaultdict(list)
    for r in rows:
        by[r["prec"]].append(r)
    for prec in sorted(by, key=lambda p: porder.get(p, 99)):
        rs = [r for r in by[prec] if r["n"] >= 2]
        diff = [r for r in rs if r["spread"]]
        out.append(f"| {prec} | {len(rs)} | {len(diff)} | {sum(r['cpu_gpu_split'] for r in diff)} | "
                   f"{sum(warn_spread(r) for r in rs)} | {sum(warn_vsd(r) for r in by[prec])} |")
    warn = [r for r in rows if warn_spread(r) or warn_vsd(r)]
    out.append(f"\n## 注意（{len(warn)} 組）\n")
    if warn:
        out.append("| 精度 | ケース | np | " + " | ".join(plats) + " | 幅 | double 比 max | CPU/GPU で分かれる |")
        out.append("|---|---|---:|" + "---:|" * len(plats) + "---:|---:|:-:|")
        for r in warn:
            sp = "-" if r["spread"] is None else f"{r['spread']:.0%}"
            vd = "-" if r["vs_double_max"] is None else f"{r['vs_double_max']:.2f}"
            out.append(f"| {r['prec']} | `{r['case']}` | {r['np']} | " +
                       " | ".join(str(r[p]) for p in plats) + f" | {sp} | {vd} | {'○' if r['cpu_gpu_split'] else ''} |")
    text = "\n".join(out) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    else:
        print(text)
    if a.csv:
        keys = ["prec", "case", "np", "n", "spread", "vs_double_max", "cpu_gpu_split"] + plats
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"組 {len(rows)}、注意 {len(warn)}", file=sys.stderr)


if __name__ == "__main__":
    main()
