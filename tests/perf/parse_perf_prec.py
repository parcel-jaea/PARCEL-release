#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""混合精度と倍精度の比較の集計 (tests/perf/gen_perf_prec.py と gen_perf.py の結果)。

  python3 tests/perf/parse_perf_prec.py <ルート> [--dir perf_results] [--md out.md] [--csv out.csv]

<ルート>/<dir>/<plat>/ の <倍精度のラベル>_<精度>_np<N> (gen_perf_prec.py) を、同じ環境・同じ並列数の
<倍精度のラベル>_np<N> (gen_perf.py) と比べる。時間は各 3 回の最速 (parse_perf.py と同じ読み方)。
  総時間の比   = 混合精度 / 倍精度 (反復数の違いを含む収束までのコスト)
  1 反復の比   = 混合精度の 1 反復 / 倍精度の 1 反復 (1 反復あたりのコスト)
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("pp", os.path.join(HERE, "parse_perf.py"))
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

BASES = ["pcg_p4_JACOBI", "cd0.05_pgmres_m60_p4_JACOBI"]


def ratio(a, b):
    return a / b if a is not None and b else None


def main():
    ap = argparse.ArgumentParser(description="混合精度と倍精度の比較の集計")
    ap.add_argument("root")
    ap.add_argument("--dir", default="perf_results")
    ap.add_argument("--md")
    ap.add_argument("--csv")
    a = ap.parse_args()
    data = pp.collect(a.root, a.dir)
    rows = []
    for plat in sorted(data):
        d = data[plat]
        for (lab, n), r in sorted(d.items()):
            base = next((b for b in BASES if lab.startswith(b + "_")), None)
            if base is None:
                continue
            prec = lab[len(base) + 1:]
            dbl = d.get((base, n), {})
            ok = r.get("status") == "OK" and dbl.get("status") == "OK"
            rows.append(dict(plat=plat, config=base, prec=prec, dev=r.get("dev"), np=n,
                             status=r.get("status"), status_double=dbl.get("status"),
                             total=r.get("total"), total_double=dbl.get("total"),
                             its=r.get("its"), its_double=dbl.get("its"),
                             ratio_total=ratio(r.get("total"), dbl.get("total")) if ok else None,
                             ratio_per_it=ratio(r.get("ms_per_it"), dbl.get("ms_per_it")) if ok else None))
    if not rows:
        print("比べる結果が見つかりません"); sys.exit(1)
    lines = ["| 環境 | 構成 | 精度 | 装置数 | 総時間 [s] (精度 / 倍精度) | 反復数 (精度 / 倍精度) | 総時間の比 | 1 反復の比 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        st = "" if x["status"] == "OK" and x["status_double"] == "OK" else f" ({x['status']} / {x['status_double']})"
        lines.append(f"| {x['plat']} | {x['config']} | {x['prec']} | {x['dev']} | "
                     f"{pp.fmt(x['total'], 2)} / {pp.fmt(x['total_double'], 2)}{st} | {x['its']} / {x['its_double']} | "
                     f"{pp.fmt(x['ratio_total'], 3)} | {pp.fmt(x['ratio_per_it'], 3)} |")
    print("\n".join(lines))
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("# 混合精度と倍精度の比較\n\n" + "\n".join(lines) + "\n")
        print(f"\n  Markdown: {a.md}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV: {a.csv} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
