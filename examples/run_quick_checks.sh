#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# ビルド後の確認: examples/ のクイックスタートをビルドして実行し、合否を 1 行で出す
#   (実行は 1 分かからない。初回はビルドに数分。2 回目からはビルド済みのものを実行するだけ)。
#
#   使い方 (パッケージのルートで。build_examples.sh と同じく SYCL 処理系と MPI の環境を整えてから):
#     export OMPI_CXX=acpp            # Open MPI の場合
#     bash examples/run_quick_checks.sh
#     PARCEL_MPI=0 bash examples/run_quick_checks.sh      # MPI なしのビルド
#     bash examples/run_quick_checks.sh --rebuild          # ビルドし直す (REBUILD=1 でも同じ)
#   最後に "PASS n / FAIL m (SKIP k)" を出し、FAIL が 0 なら終了コード 0、1 件でもあれば 1。
#
#   行うこと:
#     1. "bash examples/build_examples.sh quick" でクイックスタートの 3 本をビルド (並列。ビルドの環境変数 SYCL_TARGET・
#        CXXFLAGS・LAPACK_INC・LAPACK_LIB・MPICXX / MPICC / MPIFC・FC_LINK・WRAPPER_FLAGS・PARCEL_MPI・BUILD_JOBS は
#        そのまま build_examples.sh に渡る)。examples/build/ に 3 本がそろい、前回のビルドの記録
#        (examples/build/build_stamp.txt: 上の設定、コンパイラの場所、ヘッダ・API・サンプルのソースの cksum) が今と同じなら
#        ビルドを飛ばす。違えば違う項目を表示して作り直す
#     2. クイックスタート 4 本 (C++ quick_start・C quick_start_c・Fortran quick_start_f・Python quick_start.py) を 1 プロセスで実行
#     3. MPI ありのビルドなら C++ の quick_start を ${MPIEXEC} ${MPIEXEC_NP_FLAG} ${MPI_NP} で実行
#   各実行は「終了コード 0」かつ「出力に check: ... PASS の行がある」で合格 (1 次元ラプラシアン n=1000 の厳密解との
#   最大相対誤差が 1e-6 以下。判定の式はクイックスタートのソースのコメント)。
#   Python は parcel パッケージ (と numpy・scipy) を import できなければ SKIP (不合格にしない)。
#
#   環境変数で変えられるもの (括弧内は既定値):
#     REBUILD         1 にすると記録が同じでもビルドし直す (--rebuild と同じ)
#     NO_BUILD        1 にすると記録を見ずに、既にある examples/build/ をそのまま実行する
#                     (ログインノードでビルドし、ソースやコンパイラの見え方が違う計算ノードで実行だけするとき)
#     MPIEXEC         MPI の起動コマンド (mpirun)。Intel MPI・富士通 MPI では mpiexec
#     MPIEXEC_NP_FLAG プロセス数の指定 (-n)
#     MPI_NP          MPI の試験のプロセス数 (2)
#     MPIEXEC_ARGS    起動コマンドの追加の引数 (なし)。例: Open MPI でコア数が足りないとき --oversubscribe。
#                     MPIEXEC が Open MPI で、ここに --bind-to・--map-by が無ければ --bind-to none を足す
#                     (Open MPI は 2 ランク以下だと各ランクを 1 コアに固定し、複数スレッドが 1 コアに詰め込まれるため)
#     MPIEXEC_ALL     1 にすると 1 プロセスの実行も "${MPIEXEC} ${MPIEXEC_NP_FLAG} 1" で起動する
#                     (MPI ありのビルドで、起動コマンドを通さないと MPI_Init できない MPI (富士通 MPI など) の場合)
#     RC_ONLY         1 にすると check の行が出力に無くても終了コード 0 なら合格とする
#                     (富士通の mpiexec はランクの標準出力をジョブのログに出し、ここで受け取れないため)。
#                     未設定で MPIEXEC が富士通の mpiexec なら 1 (RC_ONLY=0 で check の行も判定する)
#     OMP_NUM_THREADS 1 プロセスあたりのスレッド数。未設定なら、使える論理 CPU 数 (CPU の割り当て (nproc)・
#                     cgroup の CPU の上限・ジョブのこのノードの割り当て (PBS の NCPUS、Slurm の SLURM_CPUS_ON_NODE) の
#                     最も小さいもの) を、計算ノード (ジョブの中) ではすべて使う。ログインノードでは 4 (使える
#                     論理 CPU が 4 未満ならその数)。MPI の実行ではそれを MPI_NP で割る (最低 1。合計がすべて / 4 になる)。
#                     AdaptiveCpp の CPU (OpenMP) のバックエンドはこの値を使う。未設定のままだと全コアぶんの
#                     スレッドを起こし、ログインノードでは他の利用者と取り合って止まったようになる。
#                     ジョブではスケジューラが設定していることがある (PBS の ompthreads・ncpus、富士通 PJM の --omp)
#     QUICK_NODE      compute / login。計算ノードかログインノードかを強制する。未設定なら、ジョブの環境変数
#                     (PBS: PBS_JOBID、富士通 PJM: PJM_JOBID、Slurm: SLURM_JOB_ID) があれば計算ノード
#     PYTHON          Python のコマンド (python3)。ネイティブ層を直接ビルドした場合は PYTHONPATH=python と
#                     PARCEL_PY_LIB=<libparcel_py.so> を与えて実行する (python/README.md)
#   ログは examples/build/run_*.log。
set -u
cd "$(dirname "$0")/.." || exit 1
REBUILD=${REBUILD:-0}
for a in "$@"; do
  case "$a" in
    --rebuild) REBUILD=1 ;;
    *) echo "使い方: bash examples/run_quick_checks.sh [--rebuild]" >&2; exit 2 ;;
  esac
done
PARCEL_MPI=${PARCEL_MPI:-1}
MPIEXEC=${MPIEXEC:-mpirun}
MPIEXEC_NP_FLAG=${MPIEXEC_NP_FLAG:--n}
MPI_NP=${MPI_NP:-2}
MPIEXEC_ARGS=${MPIEXEC_ARGS:-}
MPIEXEC_ALL=${MPIEXEC_ALL:-0}
PYTHON=${PYTHON:-python3}
B=examples/build
T0=$(date +%s)
NPASS=0; NFAIL=0; NSKIP=0; FAILED=""

pass () { NPASS=$((NPASS+1)); echo "  [PASS] $1 ($2 s)"; }
fail () { NFAIL=$((NFAIL+1)); FAILED="${FAILED} $1"; echo "  [FAIL] $1 ($2 s) $3"; }
skip () { NSKIP=$((NSKIP+1)); echo "  [SKIP] $1 $2"; }

# 1 本を実行して判定する。$1=名前 $2...=コマンド
run_check () {
  local NAME=$1; shift
  local LOG=${B}/run_${NAME}.log t=$(date +%s) rc
  "$@" < /dev/null > "${LOG}" 2>&1; rc=$?
  t=$(( $(date +%s) - t ))
  local CHECK; CHECK=$(grep -m1 '^check:' "${LOG}")
  [ -n "${CHECK}" ] && echo "         ${CHECK}"
  if [ ${rc} -ne 0 ]; then
    fail "${NAME}" ${t} "(終了コード ${rc}, ${LOG})"; tail -3 "${LOG}" | sed 's/^/         /'
  elif echo "${CHECK}" | grep -q ' PASS$'; then
    pass "${NAME}" ${t}
  elif [ -z "${CHECK}" ] && [ "${RC_ONLY}" = 1 ]; then
    pass "${NAME}" ${t}; echo "         (check の行なし。RC_ONLY=1 なので終了コードで判定)"
  else
    fail "${NAME}" ${t} "(check の PASS の行が無い, ${LOG})"
  fi
}

# 使える論理 CPU 数: CPU の割り当て (nproc。ジョブの cpuset もここに出る)・cgroup v2 の CPU の上限 (cpu.max。上の階層も)・
#   ジョブのこのノードの割り当て (PBS: NCPUS、Slurm: SLURM_CPUS_ON_NODE) のうち最も小さいもの
avail_cpus () {
  local n q p rel
  n=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
  # 自分の cgroup から上へたどり、最も小さい上限を取る (ログインノードの利用者ごとの上限は上の階層にある)
  rel=$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null | head -1)
  if [ -n "${rel}" ]; then
    rel=${rel%/}
    while :; do
      if [ -r "/sys/fs/cgroup${rel}/cpu.max" ]; then
        read -r q p < "/sys/fs/cgroup${rel}/cpu.max"
        if [ "${q}" != max ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
          q=$(( (q + p - 1) / p )); [ "${q}" -lt "${n}" ] && n=${q}
        fi
      fi
      [ -z "${rel}" ] && break
      rel=${rel%/*}
    done
  fi
  for q in "${NCPUS:-}" "${SLURM_CPUS_ON_NODE:-}"; do
    case "${q}" in ''|*[!0-9]*) ;; *) [ "${q}" -gt 0 ] && [ "${q}" -lt "${n}" ] && n=${q} ;; esac
  done
  [ "${n}" -ge 1 ] 2>/dev/null || n=1
  echo "${n}"
}
# 実行場所: QUICK_NODE (compute / login) か、ジョブの環境変数があれば計算ノード、無ければログインノード
if [ -n "${QUICK_NODE:-}" ]; then
  case "${QUICK_NODE}" in compute|login) ;; *) echo "QUICK_NODE は compute か login (${QUICK_NODE})" >&2; exit 2 ;; esac
  NODE=${QUICK_NODE}; NODE_WHY="QUICK_NODE=${QUICK_NODE}"
elif [ -n "${PBS_JOBID:-}" ]; then NODE=compute; NODE_WHY="PBS_JOBID=${PBS_JOBID}"
elif [ -n "${PJM_JOBID:-}" ]; then NODE=compute; NODE_WHY="PJM_JOBID=${PJM_JOBID}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then NODE=compute; NODE_WHY="SLURM_JOB_ID=${SLURM_JOB_ID}"
else NODE=login; NODE_WHY="ジョブの環境変数 (PBS_JOBID・PJM_JOBID・SLURM_JOB_ID) なし"
fi
[ "${NODE}" = compute ] && NODE_TXT="計算ノード" || NODE_TXT="ログインノード"
# スレッド数: OMP_NUM_THREADS が設定されていればそのまま。無ければ、計算ノードでは使える論理 CPU 数のすべて、
#   ログインノードでは 4 (使える論理 CPU が 4 未満ならその数)。MPI の実行はランク数で割る
if [ -n "${OMP_NUM_THREADS:-}" ]; then
  THR1=${OMP_NUM_THREADS}; THRN=${OMP_NUM_THREADS}
  THR_NOTE="OMP_NUM_THREADS=${OMP_NUM_THREADS} (設定済みの値)"
else
  NCPU=$(avail_cpus)
  if [ "${NODE}" = compute ]; then TOT=${NCPU}; HOW="使える論理 CPU ${NCPU} のすべて"
  else TOT=4; [ "${NCPU}" -lt 4 ] && TOT=${NCPU}; HOW="ログインノードでは 4 (使える論理 CPU ${NCPU}。4 未満ならその数)"; fi
  [ "${TOT}" -lt 1 ] && TOT=1
  THR1=${TOT}
  THRN=$(( TOT / MPI_NP )); [ "${THRN}" -lt 1 ] && THRN=1
  THR_NOTE="OMP_NUM_THREADS=${THR1} (既定: ${HOW})"
  [ "${PARCEL_MPI}" != 0 ] && THR_NOTE="${THR_NOTE}、MPI ${MPI_NP} プロセスでは ${THRN} (${TOT} を ${MPI_NP} で割った値。最低 1)"
fi
# 富士通の mpiexec (Odyssey など): ランクの標準出力がジョブのログに出て check の行を読めないので、RC_ONLY の既定を 1 に
#   (判定: MPIEXEC の名前が mpiexec で、その実体が富士通のコンパイラ・MPI の場所 (/opt/FJSV...) にある)
is_fujitsu_mpiexec () {
  local p
  [ "$(basename "${MPIEXEC}")" = mpiexec ] || return 1
  p=$(command -v "${MPIEXEC}" 2>/dev/null) || return 1
  case "$(readlink -f "${p}" 2>/dev/null || echo "${p}")" in */FJSV*) return 0 ;; esac
  return 1
}
RC_NOTE=""
if [ -z "${RC_ONLY:-}" ]; then
  RC_ONLY=0
  if [ "${PARCEL_MPI}" != 0 ] && is_fujitsu_mpiexec; then
    RC_ONLY=1; RC_NOTE="RC_ONLY=1 (既定: 富士通の mpiexec。RC_ONLY=0 で check の行も判定する)"
  fi
fi
# Open MPI: 2 ランク以下だと各ランクを 1 コアに固定するので、配置の指定が無ければ固定しない
#   (富士通の mpiexec も --version に Open MPI と出すが、配置は富士通の方法で決まるので対象外)
if [ "${PARCEL_MPI}" != 0 ] && ! is_fujitsu_mpiexec && ! echo " ${MPIEXEC_ARGS} " | grep -q -- ' --\(bind-to\|map-by\)' \
   && "${MPIEXEC}" --version < /dev/null 2>&1 | grep -qi 'open mpi\|open-rte\|openrte' \
   && ! "${MPIEXEC}" --version < /dev/null 2>&1 | grep -qi fujitsu; then
  MPIEXEC_ARGS="${MPIEXEC_ARGS:+${MPIEXEC_ARGS} }--bind-to none"
fi

# 1 プロセスの実行の前置き (MPI ありで MPIEXEC_ALL=1 のときだけ起動コマンドを通す)
L1=""
if [ "${PARCEL_MPI}" != 0 ] && [ "${MPIEXEC_ALL}" = 1 ]; then
  L1="${MPIEXEC} ${MPIEXEC_NP_FLAG} 1 ${MPIEXEC_ARGS}"
fi

echo "=== PARCEL のクイックスタートの確認 (PARCEL_MPI=${PARCEL_MPI}) ==="
echo "--- 実行場所: ${NODE_TXT} (${NODE_WHY})"
echo "--- スレッド数: ${THR_NOTE}"
[ -n "${RC_NOTE}" ] && echo "--- ${RC_NOTE}"
# ビルドが要るか: 実行ファイルがそろい、記録 (targets の行を除く) が今の設定・ソースと同じなら要らない
WHY=""
if [ "${NO_BUILD:-0}" = 1 ]; then
  echo "--- ビルドしない (NO_BUILD=1)。${B}/ の実行ファイルをそのまま使う"
else
  if [ "${REBUILD}" = 1 ]; then WHY="--rebuild"
  else
    for f in quick_start quick_start_c quick_start_f; do [ -x ${B}/${f} ] || WHY="${B}/${f} が無い"; done
    if [ -z "${WHY}" ]; then
      if [ ! -f ${B}/build_stamp.txt ]; then WHY="ビルドの記録 ${B}/build_stamp.txt が無い"
      else
        NOW=$(bash examples/build_examples.sh --print-stamp quick 2>/dev/null | grep -v '^targets=')
        DIFF=$(diff <(grep -v '^targets=' ${B}/build_stamp.txt) <(echo "${NOW}") | grep '^>' | sed 's/^> //; s/=.*//' | tr '\n' ' ')
        [ -n "${DIFF}" ] && WHY="前回のビルドと違う: ${DIFF}"
      fi
    fi
  fi
  if [ -z "${WHY}" ]; then
    echo "--- ビルドを飛ばす (${B}/ のビルドが今の設定・ソースと同じ。作り直すには --rebuild)"
  else
    echo "--- ビルド (examples/build_examples.sh quick): ${WHY}"
    t=$(date +%s)
    if bash examples/build_examples.sh quick; then pass build $(( $(date +%s) - t ))
    else fail build $(( $(date +%s) - t )) "(上のビルドの失敗を参照)"; fi
  fi
fi

echo "--- 実行 (1 プロセス, OMP_NUM_THREADS=${THR1})"
export OMP_NUM_THREADS=${THR1}
for f in quick_start quick_start_c quick_start_f; do
  if [ -x ${B}/${f} ]; then run_check ${f} ${L1} ./${B}/${f}
  else fail ${f} 0 "(${B}/${f} が無い)"; fi
done
# Python: parcel を import できなければ SKIP
PYERR=$(${L1} ${PYTHON} -c "import numpy, scipy, parcel" < /dev/null 2>&1 | tail -1)
if ${L1} ${PYTHON} -c "import numpy, scipy, parcel" < /dev/null > /dev/null 2>&1; then
  echo "         python: $(${L1} ${PYTHON} -c 'import parcel; print(parcel.__file__, parcel.__version__, "has_mpi", parcel.has_mpi())' < /dev/null 2>&1 | tail -1)"
  run_check quick_start.py ${L1} ${PYTHON} examples/quick_start.py
else
  skip quick_start.py "(${PYTHON} で parcel・numpy・scipy を import できない: ${PYERR})"
fi

if [ "${PARCEL_MPI}" != 0 ]; then
  echo "--- 実行 (MPI ${MPI_NP} プロセス: ${MPIEXEC} ${MPIEXEC_NP_FLAG} ${MPI_NP} ${MPIEXEC_ARGS}, OMP_NUM_THREADS=${THRN})"
  export OMP_NUM_THREADS=${THRN}
  if [ -x ${B}/quick_start ]; then
    run_check quick_start_np${MPI_NP} ${MPIEXEC} ${MPIEXEC_NP_FLAG} ${MPI_NP} ${MPIEXEC_ARGS} ./${B}/quick_start
  else fail quick_start_np${MPI_NP} 0 "(${B}/quick_start が無い)"; fi
fi

echo "=== PASS ${NPASS} / FAIL ${NFAIL} (SKIP ${NSKIP})  所要 $(( $(date +%s) - T0 )) s${FAILED:+  不合格:${FAILED}}"
exit $(( NFAIL > 0 ))
