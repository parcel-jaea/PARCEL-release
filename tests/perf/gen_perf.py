#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""性能試験のジョブスクリプトの生成 (PARCEL v1.2 の収束までの性能)。

  python3 tests/perf/gen_perf.py [--nx 512] [<plat> ...]    # tests/perf/sub_perf_<ソルバ>_d<装置数>_<plat>.sh

回帰試験 (tests/regression/) が「正しさ」を見るのに対し、こちらは「速さ」を見る。
実行ファイルは回帰試験と同じ bin/<plat>/verify_double を使う (tests/regression/sub_build_<plat>.sh が作る)。
コンパイル時の NX は実行時オプション PARCEL_NX を与えないときの既定値にしか使われないので、
回帰試験用 (-DNX=8) の実行ファイルをそのまま NX=512 の測定に使える。
環境ごとの設定 (キュー・資源・並列配置・walltime) は tests/site_config.py の PERF にある。

測定条件:
  問題    3 次元 7 点の格子 NX^3 (既定 NX=512、n = 134,217,728)、b = 1、x0 = 0、倍精度
  行列    非対称向けのソルバ (pgmres、pcagmres) は移流拡散 (PARCEL_MATRIX=convdiff PARCEL_CONVDIFF_C=0.05)、
          pcg・pbicgstab と固有値ソルバ (power、inverse_power、ks_arnoldi) はポアソン。
          pbicgstab は移流拡散の前処理なし NX=512 で、丸め誤差の増幅により破綻する (手法の性質) のでポアソンで測る
  収束    相対残差 1e-8 まで (最大 10,000 反復。power は 20,000、inverse_power は外側 200・内部 1e-10 / 5,000、
          ks_arnoldi は m=40・k_keep=10・nev=1・再始動 200)
          ks_arnoldi の直交化はライブラリの既定 (CGS2)。ただし PS-A (ps_a) は MGS2 (PARCEL_KS_QR=mgs。検証サンプルが
          set_qr_algo(QR_MGS) を呼ぶ。site_config.py の PERF の job_args)
  設定    ソルバのパラメータはライブラリの既定 (PARCEL_LIB_DEFAULTS=1)。明示するのは上の値と、pgmres の再始動
          m=60、pcagmres の s=t=8 と Q の作り方 (explicit / implicit)、前処理 (なし / ヤコビ)、
          inverse_power の内部ソルバ (pcg・ヤコビ前処理。ライブラリの既定の pgmres は再始動 30 のため、ポアソンでは
          最初の外側反復で内部が上限 5,000 に達して打ち切られ、pcg の 10〜14 倍の時間がかかる。外側の反復数は同じ)
  回数    各構成 3 回 (集計は最速の回)
  装置数  4 / 8 / 16 (CPU はソケット、GPU は 1 台)。1 装置あたりのランク数とスレッド数は site_config.py の PERF
出力: perf_results/<plat>/<構成>_np<N>/ (NX が 512 でなければ perf_results_nx<NX>/)。tests/perf/parse_perf.py で集計する。
"""
import argparse
import glob
import importlib.util
import os
import stat
import sys

sys.dont_write_bytecode = True       # パッケージの中に __pycache__ を作らない
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from site_config import PERF as PLATFORMS, ENV_SETUP, PROJECT, S86_CUDA_MODULE, S86_CUDA_ROOT  # noqa: E402

# PIN_SETUP (Intel MPI のランク固定) と GPU_BIND は回帰試験と共有する
_spec = importlib.util.spec_from_file_location(
    "genreg", os.path.join(HERE, "..", "regression", "gen_regression.py"))
_genreg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_genreg)
GPU_BIND = _genreg.GPU_BIND

VERIFY_SRC = "tests/verify/parcel_krylov_sample_verify.cpp"

# ---------------------------------------------------------------- 測定条件
NX = 512
DEVS = [4, 8, 16]
REPEAT = 3
TOL = "1.0e-8"
CDC = "0.05"
CDP = f"cd{CDC}_"                       # 移流拡散の構成のラベルの頭
MAT_P = "PARCEL_MATRIX=poisson"
MAT_C = f"PARCEL_MATRIX=convdiff PARCEL_CONVDIFF_C={CDC}"
CD_SOLVERS = ("pgmres", "pcagmres_explicit", "pcagmres_implicit")
# 全構成に付ける実行時オプション。PARCEL_MEM_REPORT=1 はメモリのピークも出力に残す
COMMON = "PARCEL_LIB_DEFAULTS=1 PARCEL_VERIFY=0 PARCEL_MEM_REPORT=1"
LIN = f"PARCEL_TOL={TOL} PARCEL_MAX_ITER=10000"
JAC = "jacobi_iter = 1 (default)"
# 行列のゆるい照合 (NX=512 のときだけ。外れても測定は止めず param_check.txt に soft-NG を残す)。
#   性能測定 (PARCEL_VERIFY=0) は行列名を出力しないので、残差の履歴の先頭 4 桁で移流拡散かポアソンかを見る。
#   pcagmres implicit は残差の履歴が環境で揺れるので照合しない。
SOFT_NX512 = {
    "pgmres_m60_p0_NONE": ["SOFTRE:^ +60 : residual norm = 8\\.559"],                 # ポアソンなら 8.0207e-01
    "pgmres_m60_p4_JACOBI": ["SOFTRE:^ +60 : residual norm = 7\\.673"],               # ポアソンなら 6.4587e-01
    "pcagmres_explicit_s8t8_p0_NONE": ["SOFTRE:^ +64 : residual norm = 8\\.494"],     # ポアソンなら 7.9044e-01
    "pcagmres_explicit_s8t8_p4_JACOBI": ["SOFTRE:^ +64 : residual norm = 7\\.557"],   # ポアソンなら 6.2574e-01
    "pbicgstab_p0_NONE": ["SOFTRE:^ +1 : residual norm = 4\\.1545"],                 # ポアソンの値
    "pbicgstab_p4_JACOBI": ["SOFTRE:^ +1 : residual norm = 3\\.5928"],
}


def lin_cfg(sol, p):
    """線形ソルバの 1 構成: (ラベル, 実行時オプション, 出力に現れるべき文字列)。移流拡散の構成はラベルの頭に cd0.05_"""
    pn = {0: "NONE", 4: "JACOBI"}[p]
    name = sol.split("_")[0]
    base = f"{MAT_C if sol in CD_SOLVERS else MAT_P} PARCEL_SOLVER={name} PARCEL_PRECON={p} {LIN}"
    exp = [f"[LIB_DEFAULTS] solver = {name} ,", "tol = 1.000e-08 (env)", "max_iteration = 10000 (env)",
           f"precon = {p}"] + ([JAC] if p == 4 else [])
    if sol == "pgmres":
        lab = f"pgmres_m60_p{p}_{pn}"
        args, exp = base + " PARCEL_GMRES_RESTART=60", exp + ["restart = 60 (env) , qr = MGS"]
    elif sol in ("pcagmres_explicit", "pcagmres_implicit"):
        q = sol.split("_")[1]
        qr = "MGS" if q == "explicit" else "CholeskyQR"
        lab = f"pcagmres_{q}_s8t8_p{p}_{pn}"
        args = base + f" PARCEL_CAGMRES_S=8 PARCEL_CAGMRES_T=8 PARCEL_CAGMRES_Q={q}"
        exp = exp + [f"s = 8 (env) , t = 8 (env) , q = {q} (env) , qr = {qr}"]
    else:
        lab, args = f"{sol}_p{p}_{pn}", base
    if NX == 512:
        exp = exp + SOFT_NX512.get(lab, [])
    if sol in CD_SOLVERS:
        lab = CDP + lab
    return lab, args, exp


def configs():
    """{ジョブ名 (ソルバ): [(ラベル, 実行時オプション, 期待), ...]}"""
    out = {}
    for sol in ["pcg", "pbicgstab", "pgmres", "pcagmres_explicit", "pcagmres_implicit"]:
        out[sol] = [lin_cfg(sol, 0), lin_cfg(sol, 4)]
    # 固有値ソルバは前処理なしだけ (前処理を付けると A M^-1 の固有値という別の問題になる)
    out["power"] = [("power_p0_NONE", f"{MAT_P} PARCEL_SOLVER=power PARCEL_PRECON=0 PARCEL_TOL={TOL} PARCEL_MAX_ITER=20000",
                     ["[LIB_DEFAULTS] solver = power ,", "tol = 1.000e-08 (env)", "max_iteration = 20000 (env)",
                      "precon = 0"])]
    out["inverse_power"] = [("inverse_power_p0_NONE",
                             f"{MAT_P} PARCEL_SOLVER=inverse_power PARCEL_PRECON=0 PARCEL_TOL={TOL} PARCEL_MAX_ITER=200 "
                             "PARCEL_INNER_TOL=1.0e-10 PARCEL_INNER_MAX=5000 PARCEL_LINSOLVE_SOLVER=pcg PARCEL_LINSOLVE_PRECON=4",
                             ["[LIB_DEFAULTS] solver = inverse_power ,", "tol = 1.000e-08 (env)",
                              "max_iteration = 200 (env)", "precon = 0", "linsolve = pcg (env)",
                              "linsolve_tol = 1.000e-10 (env)", "linsolve_max = 5000 (env)", "linsolve_precon = 4"])]
    out["ks_arnoldi"] = [("ks_arnoldi_p0_NONE",
                          f"{MAT_P} PARCEL_SOLVER=ks_arnoldi PARCEL_PRECON=0 PARCEL_KS_M=40 PARCEL_KS_K_KEEP=10 "
                          f"PARCEL_KS_NEV=1 PARCEL_TOL={TOL} PARCEL_KS_RESTART=200",
                          ["[LIB_DEFAULTS] solver = ks_arnoldi ,", "tol = 1.000e-08 (env)", "precon = 0",
                           "ks_m = 40 (env) , ks_k_keep = 10 (env) , ks_nev = 1 (env) , ks_max_restart = 200 (env)"])]
    return out


# ---------------------------------------------------------------- 並列配置・資源
def layout(cfg, dev):
    """装置数 dev のノード数・ランク数・1 ノードのランク数"""
    if cfg["kind"] == "gpu":
        return dict(nodes=-(-dev // cfg["dpn"]), np=dev, ppn=min(dev, cfg["dpn"]))
    return dict(nodes=dev // cfg["dpn"], np=dev * cfg["rpd"], ppn=cfg["rpd"] * cfg["dpn"])


def select_of(cfg, dev):
    if cfg.get("chunk_per") == "device":
        return f"select={dev}:{cfg['chunk']}"
    return f"select={layout(cfg, dev)['nodes']}:{cfg['chunk']}"


def place_of(cfg, dev):
    """PBS の配置の行 (site_config.py の place)。チャンクが 1 ノードぶん以下なら place[0]、超えれば place[1]"""
    pl = cfg.get("place")
    if not pl:
        return ""
    if cfg.get("chunk_per") == "device":
        fits = dev <= cfg["dpn"]
    else:
        fits = layout(cfg, dev)["nodes"] <= 1
    return f"#PBS -l place={pl[0] if fits else pl[1]}\n"


def hms_sec(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def walltime_of(cfg, job, dev):
    return cfg.get("walltime_by", {}).get(job, {}).get(dev, cfg["walltime"])


def queue_of(cfg, dev, wall):
    nodes = layout(cfg, dev)["nodes"]
    for maxn, maxw, q in cfg["queues"]:
        if (maxn is None or nodes <= maxn) and hms_sec(wall) <= hms_sec(maxw):
            return q
    sys.exit(f"収まるキューが無い (ノード {nodes}、walltime {wall}): {cfg['queues']}")


def launch_of(cfg, tag):
    """起動コマンド。Open MPI は --map-by で配置し、Intel MPI は -ppn と I_MPI_PIN_DOMAIN (psa_numa_pin) で固定する"""
    if cfg.get("mpi_impl") == "intel":
        return "mpirun -n ${NP} -ppn ${PPN}"
    if cfg["kind"] == "cpu":
        # 1 ソケットに rpd ランク
        return f"mpirun -n ${{NP}} --map-by ppr:{cfg['rpd']}:socket:PE=${{OMP_NUM_THREADS}} --bind-to core"
    # 1 ノードに載せるランク数は、そのノードの GPU の数を超えない (超えたランクは GPU を掴めず CPU で動き、
    #   集団通信が全体を律速する)
    return (f"mpirun -n ${{NP}} --map-by ppr:${{PPR}}:node:PE=${{OMP_NUM_THREADS}} --bind-to core "
            f"${{WORKDIR}}/gpu_bind_{tag}.sh")


def env_of(cfg):
    if cfg["sched"] == "pbs":
        cuda = cfg["gpu"]
        return ENV_SETUP[cfg["env"]].format(
            cuda_mod=f"module load {S86_CUDA_MODULE}\n" if cuda else "",
            cuda_env=(f"export cuda_root={S86_CUDA_ROOT}\nexport ACPP_CUDA_PATH=$cuda_root\n") if cuda else "",
            ld="$cuda_root/lib64:" if cuda else "", parcel_load="")
    # s86 形式以外は format しない。{parcel_load} (回帰試験の C / Fortran / Python 用) は性能試験では使わない
    return ENV_SETUP[cfg["env"]].replace("{parcel_load}", "")


def header(name, plat, cfg, dev, wall, title):
    L = layout(cfg, dev)
    q = queue_of(cfg, dev, wall)
    proj = PROJECT[cfg["project"]]
    sel = select_of(cfg, dev)
    place = place_of(cfg, dev)
    desc = (f"# 性能試験: {title} / {plat} / 装置数 {dev}\n"
            f"#   NX={NX} (n={NX**3:,})、b = 1、x0 = 0、収束まで (相対残差 {float(TOL):.0e}、最大 10,000)。"
            f"各構成 {REPEAT} 回 (集計で最速の回)\n"
            f"#   行列: pcg・pbicgstab・固有値ソルバはポアソン、pgmres・pcagmres は移流拡散 (c={CDC}、ラベルの頭 {CDP})\n"
            f"#   {cfg['note']}\n"
            f"#   np={L['np']}、{L['nodes']} ノード。生成: tests/perf/gen_perf.py。パッケージのルートから投入する\n")
    if cfg["sched"] == "pbs_miy":
        h = (f"#PBS -q {q}\n#PBS -l {sel}\n{place}#PBS -l walltime={wall}\n#PBS -W group_list={proj}\n"
             f"#PBS -N PARCEL_P_{name}\n#PBS -j oe\n#PBS -o perf_{name}.log\n")
    elif cfg["sched"] == "pbs_ps":
        h = (f"#PBS -N PARCEL_P_{name}\n#PBS -P {proj}\n#PBS -q {q}\n#PBS -l {sel}\n{place}#PBS -l walltime={wall}\n"
             f"#PBS -o perf_{name}.log\n#PBS -j oe\n")
    else:
        h = (f"#PBS -N PARCEL_P_{name}\n#PBS -P {proj}\n#PBS -q {q}\n#PBS -l {sel}\n{place}#PBS -l walltime={wall}\n"
             f"#PBS -o perf_{name}.log\n#PBS -e perf_{name}.err\n")
    return f"#!/bin/bash\n{desc}{h}\n{env_of(cfg)}\n"


BODY = """
# ======== 性能試験の本体 (tests/perf/gen_perf.py が生成) ========
cd ${{WORKDIR}} || exit 1
PLAT={plat}
OUT={outdir}/${{PLAT}}
mkdir -p ${{OUT}}
BIN=${{WORKDIR}}/{bin_rel}
if [ ! -x "${{BIN}}" ]; then
  echo "  BUILD_MISSING ${{BIN}}"
  echo "  先に {build_script} で実行ファイルを作ること"
  exit 1
fi
# PARCEL はヘッダオンリーなので、.cpp だけでなく Include/ の更新も見る (古い実行ファイルで測らない)
NEWEST=$(ls -t {verify_src} Include/*.hpp 2>/dev/null | head -1)
if [ "${{NEWEST}}" -nt "${{BIN}}" ]; then
  echo "  WARNING: ${{NEWEST}} が ${{BIN}} より新しい。ビルドし直すこと"
fi
echo "=== $(date '+%F %T') {name} ($(ls -l --time-style=+%F_%T ${{BIN}} | awk '{{print $6}}') の {bin_show})"
# 実行時オプションはコマンドライン引数で渡す (環境変数は MPI 実装によっては他のノードのランクに届かない)。
#   検証サンプルは環境変数を引数より優先するので、ジョブの環境に PARCEL_* が残っていれば知らせる
env | grep '^PARCEL_' | grep -v '^PARCEL_PERF_FORCE=' | sed 's/^/  環境変数 (引数より優先される): /'
export OMP_NUM_THREADS={omp}
# OMP_PROC_BIND は設定しない。libomp は proc_bind が有効だと初期スレッドを 1 コアに固定し、
#   それを継承した AdaptiveCpp のワーカースレッドから見えるコアが 1 つになってスレッド並列が効かなくなる。
#   ランクの固定は MPI 側で行う。
export OMP_PROC_BIND=false
unset OMP_PLACES
export PARALLEL=${{OMP_NUM_THREADS}}
NP={np}; DEV={dev}; PPN={ppn}
{ppr}TMO={tmo}                               # 1 回の打ち切り [秒] (walltime と同じ)
{gpu_bind}
@@PIN@@
# 期待する文字列の照合。$1=ファイル $2='|' 区切り
#   "SOFTRE:" は行列のゆるい照合 (正規表現): soft-ok / soft-NG を残すが、外れても止めない
chk () {{
  local f=$1 e ng=0 a
  IFS='|' read -r -a a <<< "$2"
  for e in "${{a[@]}}"; do
    if [[ "${{e}}" == SOFTRE:* ]]; then grep -qE -- "${{e#SOFTRE:}}" ${{f}} 2>/dev/null && echo "soft-ok  ${{e}}" || echo "soft-NG  ${{e}}"
    else grep -qF -- "${{e}}" ${{f}} 2>/dev/null && echo "ok  ${{e}}" || {{ echo "NG  ${{e}}"; ng=1; }}; fi
  done
  return ${{ng}}
}}

# $1=ラベル $2=期待 ('|' 区切り) $3...=実行時オプション
run_unit () {{
  local LABEL=$1 EXP=$2; shift 2
  local D=${{OUT}}/${{LABEL}}_np${{NP}} R RC TR
  # 済んだ構成は飛ばす (walltime 切れでも再投入すれば残りだけ測れる)。測り直すときは PARCEL_PERF_FORCE=1
  if [ -z "${{PARCEL_PERF_FORCE:-}}" ] && [ "$(cat ${{D}}/status.txt 2>/dev/null)" = "OK" ]; then
    echo "  [skip] ${{LABEL}} np=${{NP}}"; return
  fi
  mkdir -p ${{D}}
  rm -f ${{D}}/status.txt ${{D}}/run*.txt ${{D}}/result*.txt ${{D}}/time*.txt ${{D}}/param_check.txt ${{D}}/bindings.txt \\
        ${{D}}/parcel_result.txt ${{D}}/mem_*.txt
  echo ${{DEV}} > ${{D}}/dev.txt
  # 実際の割り当てを記録する (ランクがどのソケット・ノードに載ったかの確認用)
  {bindings}
  TR=$(date +%s)
  for R in $(seq 1 {rep}); do
    local T0=$(date +%s)
    ( cd ${{D}} && timeout -k 60 ${{TMO}} {launch} ${{BIN}} "$@" PARCEL_NX={nx} PARCEL_OUTPUT=result${{R}}.txt \\
        > run${{R}}.txt 2>&1 < /dev/null )
    RC=$?
    echo $(( $(date +%s)-T0 )) > ${{D}}/time${{R}}.txt
{pin_record}    if [ ${{RC}} -eq 124 ]; then echo TIMEOUT > ${{D}}/status.txt; break
    elif [ ${{RC}} -ne 0 ]; then echo "RUN_FAIL_${{RC}}" > ${{D}}/status.txt; break; fi
    # 1 回目の出力で、実行時オプションが効いているか (と行列) を確かめる
    if [ ${{R}} = 1 ] && [ "${{EXP}}" != - ]; then
      if ! chk ${{D}}/result1.txt "${{EXP}}" > ${{D}}/param_check.txt; then echo PARAM_NG > ${{D}}/status.txt; break; fi
      grep -q '^soft-NG' ${{D}}/param_check.txt && echo "    行列の照合 (soft) が外れた: $(grep '^soft-NG' ${{D}}/param_check.txt | head -1)"
    fi
  done
  [ -f ${{D}}/status.txt ] || echo OK > ${{D}}/status.txt
  echo "  [$(cat ${{D}}/status.txt)] ${{LABEL}} np=${{NP}}  ($(( $(date +%s)-TR )) 秒)"
}}

"""


def gen_job(job, plat, cfg, dev, units, prec="double", name=None, title=None, bin_rel=None, build_script=None):
    """1 ジョブ (1 環境・1 装置数) のスクリプトを書き、パスを返す。units = [(ラベル, 実行時オプション, 期待), ...]
    bin_rel: 実行ファイル (パッケージのルートからの相対。既定 bin/<plat>/verify_<prec>)、build_script: それを作るスクリプト"""
    bin_rel = bin_rel or f"bin/{plat}/verify_{prec}"
    build_script = build_script or f"tests/regression/sub_build_{plat}.sh"
    name = name or f"{job}_d{dev}_{plat}{SFX}"
    wall = walltime_of(cfg, job, dev)
    L = layout(cfg, dev)
    s = header(name, plat, cfg, dev, wall, title or job)
    if cfg.get("mpi_impl") == "intel":
        # Intel MPI: I_MPI_DEBUG=4 のピン表を 1 回目の出力から bindings.txt に残す
        bindings = "export I_MPI_DEBUG=4"
        pin_record = ("    if [ ${R} = 1 ]; then sed -n '/MPI startup().*Pin cpu/,/^$/p' ${D}/run1.txt > ${D}/bindings.txt; fi\n"
                      "    grep -q 'Too many pin domains' ${D}/run${R}.txt && { echo PIN_IGNORED > ${D}/status.txt; break; }\n")
    else:
        bindings = launch_of(cfg, name).replace(f" ${{WORKDIR}}/gpu_bind_{name}.sh", "") + \
            " --report-bindings /bin/true > ${D}/bindings.txt 2>&1 < /dev/null"
        pin_record = ""
    s += BODY.format(plat=plat, bin_rel=bin_rel, bin_show=bin_rel[len(f"bin/{plat}/"):], build_script=build_script, name=name, outdir=OUTDIR, verify_src=VERIFY_SRC, omp=cfg["omp"],
                     np=L["np"], dev=dev, ppn=L["ppn"],
                     ppr=(f"PPR={L['ppn']}      # 1 ノードに載せるランク数 (GPU の数を超えない)\n"
                          if cfg["kind"] == "gpu" else ""), tmo=hms_sec(wall), rep=REPEAT, nx=NX,
                     launch=launch_of(cfg, name), bindings=bindings, pin_record=pin_record,
                     gpu_bind=(GPU_BIND.replace("{tag}", name) if cfg["kind"] == "gpu" else ""))
    if cfg.get("pin"):
        pin = (_genreg.PIN_SETUP[cfg["pin"]] +
               f"\npsa_numa_pin ${{PPN}} ${{OMP_NUM_THREADS}} {cfg.get('pin_mode', 'spread')} || "
               "{ echo 'PIN_SETUP_FAIL'; exit 1; }\n")
    else:
        pin = ""
    s = s.replace("@@PIN@@", pin)
    extra = cfg.get("job_args", {}).get(job, "")
    for lab, args, exp in units:
        if extra:
            # 環境ごとの追加の実行時オプション (site_config.py の job_args)。PARCEL_KS_QR は効いたことを出力で確かめる
            args = f"{args} {extra}"
            for v in extra.split():
                if v.startswith("PARCEL_KS_QR="):
                    exp = exp + ["ks_max_restart = 200 (env) , qr = %s (env)" % v.split("=", 1)[1].upper().rstrip("2")]
        e = '"' + "|".join(exp) + '"' if exp else "-"
        s += f"run_unit {lab} {e} {COMMON} {args}\n"
    s += '\necho "=== $(date \'+%F %T\') ALL DONE ==="\n'
    path = os.path.join(HERE, f"sub_perf_{name}.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


SFX = ""                    # NX が 512 でないときの接尾辞 ("_nx256" など)
OUTDIR = "perf_results"


def set_nx(nx):
    global NX, SFX, OUTDIR
    NX = nx
    SFX = "" if nx == 512 else f"_nx{nx}"
    OUTDIR = "perf_results" + SFX


def main():
    ap = argparse.ArgumentParser(description="性能試験のジョブスクリプトを生成する")
    ap.add_argument("--nx", type=int, default=512, help="格子の一辺 (既定 512)。512 以外は出力先・名前に _nx<NX> を付ける")
    ap.add_argument("plats", nargs="*", help="生成する環境 (既定は tests/site_config.py の PERF の全環境)")
    args = ap.parse_args()
    set_nx(args.nx)
    plats = args.plats or list(PLATFORMS)
    for p in plats:
        if p not in PLATFORMS:
            sys.exit(f"未知の環境 {p} (tests/site_config.py の PERF にあるもの: {' '.join(PLATFORMS)})")
    # 同じ NX の古いスクリプトを消す (混合精度の比較 _prec_ は gen_perf_prec.py の担当)
    stale = [f for f in glob.glob(os.path.join(HERE, "sub_perf_*.sh"))
             if "_prec_" not in os.path.basename(f)
             and (os.path.basename(f).endswith(f"{SFX}.sh") if SFX else "_nx" not in os.path.basename(f))
             and any(os.path.basename(f).endswith(f"_{p}{SFX}.sh") for p in plats)]
    for f in stale:
        os.remove(f)
    if stale:
        print(f"  既存スクリプト {len(stale)} 件を削除")
    cfgs = configs()
    total = nj = 0
    for plat in plats:
        cfg = PLATFORMS[plat]
        n = 0
        for dev in DEVS:
            for job, units in cfgs.items():
                gen_job(job, plat, cfg, dev, units)
                nj += 1
                n += len(units)
        total += n
        print(f"  {plat:<13} {len(cfgs) * len(DEVS):>3} ジョブ / {n:>3} 構成  装置数 {'/'.join(map(str, DEVS))}  "
              f"np = {'/'.join(str(layout(cfg, d)['np']) for d in DEVS)}  omp = {cfg['omp']}")
    print(f"  {'合計':<13} {nj:>3} ジョブ / {total:>3} 構成 ({total * REPEAT} 実行)")


if __name__ == "__main__":
    main()
