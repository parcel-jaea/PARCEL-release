#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""混合精度 (前処理のデータを単精度にする precon_dataF など) の性能を倍精度と比べるジョブの生成。

  python3 tests/perf/gen_perf_prec.py [--prec precon_dataF[,<精度>...]] [--nx 512] [<plat> ...]

性能試験 (tests/perf/gen_perf.py) と同じ条件 (NX=512、収束まで、各 3 回、装置数 4 / 8 / 16、同じ並列配置) で、
実行ファイルだけを精度構成ごとの bin/<plat>/perf_prec/verify_<精度> に替えて測る。
実行ファイルはこの生成器が書くビルドスクリプト tests/perf/sub_build_prec_<plat>.sh が作る (回帰試験のビルドジョブ
tests/regression/sub_build_<plat>.sh と同じ環境・コンパイル行・精度フラグに PERF_EXTRA_FLAGS を足したもの)。
  precon_dataF は Jacobi 前処理のその場計算なし (-DPARCEL_JACOBI_ONTHEFLY=0) で作る。既定 (省メモリ設定 B、
  PARCEL_JACOBI_ONTHEFLY=1) では前処理の行列を float で持っても、前処理のたびに元の行列 A の double の値を読むので
  混合精度の利点が出にくい。その場計算なしでは前処理の行列 (D^-1)(L+U) を float で保存して読む
  (そのぶんメモリが増える: NX=512 の 7 点の格子で 1 行あたり約 28 B、装置数 4 で 1 台あたり約 0.9 GB)。
  比べる相手の倍精度 (bin/<plat>/verify_double) と回帰試験の precon_dataF (bin/<plat>/verify_precon_dataF) は既定のまま。
  前処理後のマクロ (PARCEL_JACOBI_ONTHEFLY) は bin/<plat>/perf_prec/macro_<精度>.txt に残る。
構成はヤコビ前処理の 2 つ: pcg (ポアソン) と pgmres m=60 (移流拡散)。比べる相手の倍精度は gen_perf.py の
同じラベルの構成 (pcg_p4_JACOBI と cd0.05_pgmres_m60_p4_JACOBI)。
ラベルは <倍精度のラベル>_<精度>、出力先は性能試験と同じ perf_results/<plat>/ (tests/perf/parse_perf_prec.py で集計する)。
スクリプト: tests/perf/sub_build_prec_<plat>.sh (ビルド。ログインノードで bash で実行してよい。A64FX は計算ノードで投入)、
            tests/perf/sub_perf_d<装置数>_<plat>_prec_<精度>.sh (1 ジョブ = 1 環境 x 1 装置数 x 1 精度)
"""
import argparse
import glob
import importlib.util
import os
import re
import stat
import sys

sys.dont_write_bytecode = True       # パッケージの中に __pycache__ を作らない
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from site_config import PERF_PREC_PLATS  # noqa: E402

_spec = importlib.util.spec_from_file_location("gp", os.path.join(HERE, "gen_perf.py"))
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


# 性能試験だけで精度フラグに足すもの (回帰試験の PRECISIONS は変えない)。
#   precon_dataF: Jacobi 前処理のその場計算なし (2026-10-09)。前処理の行列を float で保存して読む
PERF_EXTRA_FLAGS = {"precon_dataF": "-DPARCEL_JACOBI_ONTHEFLY=0"}
SUBDIR = "perf_prec"                  # 実行ファイルの置き場 bin/<plat>/perf_prec/ (回帰試験の bin/<plat>/verify_* と分ける)


def prec_flags(prec):
    return " ".join(x for x in (gp._genreg.PRECISIONS[prec]["flags"], PERF_EXTRA_FLAGS.get(prec, "")) if x)


def build_script(plat, precs):
    """回帰試験のビルドジョブ (gen_regression.py の build_job) から、出力先を bin/<plat>/perf_prec/ に替え、
    作るものを precs だけ (精度フラグ + PERF_EXTRA_FLAGS) にしたスクリプトを返す"""
    gr = gp._genreg
    s = gr.build_job(plat, gr.PLATFORMS[plat])
    lines = s.split("\n")
    assert lines[1].startswith("# 回帰試験・性能試験の実行ファイルを作る: "), (plat, lines[1])
    i = 2
    while lines[i].startswith("#   "):
        i += 1
    desc = [f"# 混合精度の性能試験の実行ファイルを作る: {plat} (tests/perf/gen_perf_prec.py が生成)",
            f"#   {' '.join(precs)} を bin/{plat}/{SUBDIR}/verify_<精度> に作る。精度フラグは回帰試験と同じに PERF_EXTRA_FLAGS を足す"]
    desc += [f"#     {p}: {prec_flags(p)}" for p in precs]
    desc += ["#   環境・コンパイル行は tests/regression/sub_build_" + plat + ".sh と同じ。前処理後のマクロを macro_<精度>.txt に残す。",
             "#   出来ている実行ファイルは飛ばす。作り直すときは消してから。"]
    s = "\n".join(lines[i:])
    for old, new in ((f"PARCEL_B_{plat}", f"PARCEL_BP_{plat}"), (f"build_{plat}.", f"build_prec_{plat}."),
                     (f"bin/{plat}", f"bin/{plat}/{SUBDIR}"), (f"buildlog/{plat}", f"buildlog/{plat}/{SUBDIR}")):
        assert old in s, (plat, old)
        s = s.replace(old, new)
    s = s.replace(f"tests/regression/sub_build_prec_{plat}.sh", f"tests/perf/sub_build_prec_{plat}.sh")
    s = "\n".join(lines[:1] + desc) + "\n" + s
    # 回帰試験の作るもの (全精度と ddcheck) を外し、precs だけを作る
    t = "T_ALL=$(date +%s)\n"
    assert s.count(t) == 1, plat
    s = s[:s.index(t) + len(t)]
    # 前処理後のマクロ: build_one のコンパイル行に -dM -E を付けたもの
    m = re.search(r'^\s*(\$\{CXX\} -O3 .*?"\$@" \$\{EXTRA\} .*?) \\\n\s*-o \$\{OUT\} (\S+) ', s, re.M)
    assert m, plat
    src = m.group(2)
    s = s.replace(t, "macro_one () {   # $1=精度 $2...=精度フラグ\n"
                     "  P=$1; shift\n"
                     f"  {m.group(1)} -dM -E {src} 2> /dev/null \\\n"
                     "    | grep -E 'define PARCEL_(JACOBI_ONTHEFLY|SPMV_LADDER|PRECON_LADDER) ' | sort -u > ${BINDIR}/macro_${P}.txt\n"
                     "  echo \"  マクロ ${P}: $(tr '\\n' ' ' < ${BINDIR}/macro_${P}.txt)\"\n"
                     "}\n" + t)
    for p in precs:
        s += ('while [ "$(jobs -rp | wc -l)" -ge ${NPAR} ]; do wait -n 2>/dev/null || sleep 2; done\n'
              f'build_one {p} {prec_flags(p)} &\n')
    s += "wait\n"
    for p in precs:
        s += f"macro_one {p} {prec_flags(p)}\n"
    s += 'echo "  ビルド全体 $(( $(date +%s)-T_ALL )) 秒"\n'
    s += '\necho "=== ALL DONE ==="\n'
    return s


def units_for(prec):
    out = []
    for sol in ("pcg", "pgmres"):
        lab, args, exp = gp.lin_cfg(sol, 4)
        out.append((f"{lab}_{prec}", args, exp))
    return out


def main():
    ap = argparse.ArgumentParser(description="混合精度と倍精度の比較のジョブスクリプトを生成する")
    ap.add_argument("--prec", default="precon_dataF",
                    help="比べる精度構成 (カンマ区切り。tests/regression/gen_regression.py の PRECISIONS の名前)")
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("plats", nargs="*", help="環境 (既定は tests/site_config.py の PERF_PREC_PLATS)")
    args = ap.parse_args()
    gp.set_nx(args.nx)
    precs = [p for p in args.prec.split(",") if p]
    for p in precs:
        if p == "double" or p not in gp._genreg.PRECISIONS:
            sys.exit(f"未知の精度 {p} (double 以外の {' '.join(gp._genreg.PRECISIONS)})")
    plats = args.plats or PERF_PREC_PLATS
    for p in plats:
        if p not in gp.PLATFORMS:
            sys.exit(f"未知の環境 {p} (tests/site_config.py の PERF)")
    for f in glob.glob(os.path.join(HERE, f"sub_perf_*_prec_*{gp.SFX}.sh")):
        if any(f"_{p}_prec_" in os.path.basename(f) for p in plats):
            os.remove(f)
    for plat in plats:
        if plat not in gp._genreg.PLATFORMS:
            sys.exit(f"環境 {plat} が回帰試験の設定 (tests/site_config.py の REGRESSION) に無いのでビルドスクリプトを作れない")
        path = os.path.join(HERE, f"sub_build_prec_{plat}.sh")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_script(plat, precs))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    n_jobs = n_units = 0
    for plat in plats:
        cfg = gp.PLATFORMS[plat]
        for prec in precs:
            for dev in gp.DEVS:
                us = units_for(prec)
                gp.gen_job("prec", plat, cfg, dev, us, prec=prec, name=f"d{dev}_{plat}_prec_{prec}{gp.SFX}",
                           title=f"混合精度 {prec} (pcg ヤコビ: ポアソン、pgmres ヤコビ: 移流拡散)",
                           bin_rel=f"bin/{plat}/{SUBDIR}/verify_{prec}",
                           build_script=f"tests/perf/sub_build_prec_{plat}.sh")
                n_jobs += 1
                n_units += len(us)
    print(f"  {n_jobs} ジョブ / {n_units} 構成 ({n_units * gp.REPEAT} 実行)。倍精度の相手は gen_perf.py の pcg / pgmres の"
          " ヤコビ前処理の構成")
    print(f"  ビルド: tests/perf/sub_build_prec_<plat>.sh ({len(plats)} 本、bin/<plat>/{SUBDIR}/verify_<精度>)")
    for p in precs:
        print(f"    {p}: {prec_flags(p)}")


if __name__ == "__main__":
    main()
