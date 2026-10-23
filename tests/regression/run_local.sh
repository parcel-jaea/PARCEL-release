#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# スケジューラの無い計算機 (site_config.py の sched="local"、例: mac) で回帰試験をビルド・実行・集計する。
#   ジョブスクリプト (sub_regression_<plat>_*.sh) を bash でそのまま流す。中身 (ケース・判定・結果の置き場所) は
#   スケジューラのある環境と同じ。同時に流すジョブは「使うコア数 (並列数 x スレッド数) の合計が上限以下」に収める。
#
#   使い方 (パッケージのルートで):
#     bash tests/regression/run_local.sh --plat mac [--mpi yes|no|both] [--lang cpp,c,fortran,python]
#                                  [--build] [--no-run] [--cores <N>] [--dry-run]
#     --plat    環境 (カンマ区切り。MPI ありの名前を与えれば --mpi で <plat>_nompi も選ぶ)。必須
#     --mpi     yes = MPI あり、no = MPI なし、both = 両方 (既定)
#     --lang    cpp (検証サンプルの全精度 + ddcheck) / c / fortran / python。既定は全部
#     --build   先にビルドする (sub_build_<plat>.sh と sub_build_lang_<plat>.sh <言語>)。環境ごとに順に流し、
#               同時に 2 つのビルドを走らせない (メモリを倍食わないように)。同時コンパイル数は PARCEL_BUILD_PAR
#     --no-run  ビルドだけ
#     --cores   同時に使うコア数の上限 (既定: 論理 CPU 数。mac の M4 では 10)。1 ジョブがこれを超えるときは単独で流す
#     --dry-run 流すジョブの一覧だけを出す
#   ログ: tests/regression/local_logs/<plat>/<ジョブ名>.log、全体の経過は標準出力 (nohup で流すなら > ファイル)。
#   最後に tests/regression/summarize.sh を同じ選択で呼ぶ。済んだケース (status.txt が OK) は各ジョブが飛ばすので、
#   途中で止めても同じコマンドで続きから流せる (やり直すときは PARCEL_REG_FORCE=1)。
#   macOS の /bin/bash (3.2) で動く書き方にしてある (連想配列・wait -n・mapfile を使わない)。
cd "$(dirname "$0")/../.." || exit 1
R=tests/regression
PLAT=""; MPI=both; LSEL="cpp,c,fortran,python"; BUILD=0; RUN=1; DRY=0
CORES=$(getconf _NPROCESSORS_ONLN 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4)
while [ $# -gt 0 ]; do
  case $1 in
    --plat)    PLAT=$2; shift 2 ;;
    --plat=*)  PLAT=${1#--plat=}; shift ;;
    --mpi)     MPI=$2; shift 2 ;;
    --mpi=*)   MPI=${1#--mpi=}; shift ;;
    --lang)    LSEL=$2; shift 2 ;;
    --lang=*)  LSEL=${1#--lang=}; shift ;;
    --cores)   CORES=$2; shift 2 ;;
    --cores=*) CORES=${1#--cores=}; shift ;;
    --build)   BUILD=1; shift ;;
    --no-run)  RUN=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '3,21p' "$0"; exit 0 ;;
    *) echo "未知の引数 $1"; exit 2 ;;
  esac
done
[ -n "${PLAT}" ] || { echo "--plat を与えること (例: --plat mac)"; exit 2; }
case "${MPI}" in yes|no|both) ;; *) echo "--mpi は yes / no / both"; exit 2 ;; esac
LANGS=""
for L in $(echo "${LSEL}" | tr ',' ' '); do
  case $L in
    cpp|c++|cxx) L=cpp ;; c) ;; f|fortran) L=f ;; py|python) L=py ;;
    *) echo "未知の言語 $L (cpp / c / fortran / python)"; exit 2 ;;
  esac
  LANGS="${LANGS} ${L}"
done
PLATS=""
for P in $(echo "${PLAT}" | tr ',' ' '); do
  B=${P%_nompi}
  case ${MPI} in
    yes)  PLATS="${PLATS} ${B}" ;;
    no)   PLATS="${PLATS} ${B}_nompi" ;;
    both) PLATS="${PLATS} ${B} ${B}_nompi" ;;
  esac
done
has_lang () { case " ${LANGS} " in *" $1 "*) return 0 ;; esac; return 1; }
T_ALL=$(date +%s)
echo "=== run_local $(date '+%F %T') 環境:${PLATS} 言語:${LANGS} コア上限 ${CORES} (build=${BUILD} run=${RUN})"

# ---------------------------------------------------------------- ビルド (環境ごとに順に)
if [ ${BUILD} = 1 ] && [ ${DRY} = 0 ]; then
  for P in ${PLATS}; do
    mkdir -p ${R}/local_logs/${P}
    if has_lang cpp; then
      [ -f ${R}/sub_build_${P}.sh ] || { echo "${R}/sub_build_${P}.sh が無い (tests/regression/gen_regression.py で作ること)"; exit 1; }
      T0=$(date +%s)
      bash ${R}/sub_build_${P}.sh > ${R}/local_logs/${P}/build.log 2>&1 < /dev/null
      echo "  [build] ${P} C++: $(grep -cE '^ *\[(OK|skip)\]' ${R}/local_logs/${P}/build.log) 本 OK/skip," \
           "$(grep -c '^ *\[FAIL\]' ${R}/local_logs/${P}/build.log) 本 FAIL ($(( $(date +%s)-T0 )) 秒)"
      grep '^ *\[FAIL\]' ${R}/local_logs/${P}/build.log | sed 's/^/    /'
    fi
    LL=""
    for L in c f py; do has_lang $L && LL="${LL} $L"; done
    if [ -n "${LL}" ]; then
      [ -f ${R}/sub_build_lang_${P}.sh ] || { echo "${R}/sub_build_lang_${P}.sh が無い"; exit 1; }
      T0=$(date +%s)
      bash ${R}/sub_build_lang_${P}.sh ${LL} > ${R}/local_logs/${P}/build_lang.log 2>&1 < /dev/null
      echo "  [build] ${P} 言語別 (${LL# }): 作れた言語 $(tr '\n' ' ' < bin/${P}/lang_built.txt 2> /dev/null)($(( $(date +%s)-T0 )) 秒)"
      grep -E '\[FAIL\]|MISSING' ${R}/local_logs/${P}/build_lang.log | sed 's/^/    /'
    fi
  done
fi
[ ${RUN} = 1 ] || exit 0

# ---------------------------------------------------------------- 流すジョブの一覧
JOBS=""
for P in ${PLATS}; do
  for f in ${R}/sub_regression_${P}_*.sh; do
    [ -f "$f" ] || continue
    rest=${f#${R}/sub_regression_${P}_}
    case ${rest} in
      nompi_*) continue ;;                                   # <plat>_nompi のスクリプト (別の環境)
      lang_c_*) has_lang c || continue ;;
      lang_f_*) has_lang f || continue ;;
      lang_py_*) has_lang py || continue ;;
      lang_*) continue ;;
      *) has_lang cpp || continue ;;
    esac
    JOBS="${JOBS} ${P}:$f"          # 環境名:スクリプト
  done
done
NJ=$(echo ${JOBS} | wc -w | tr -d ' ')
[ "${NJ}" -gt 0 ] || { echo "流すジョブが無い"; exit 1; }
# 使うコア数 = 並列数 (ファイル名の _np<N>) x スレッド数 (スクリプトの OMP_NUM_THREADS)
weight () {
  local np omp
  np=$(echo "$1" | sed -E 's/.*_np([0-9]+)\.sh$/\1/')
  omp=$(grep -m1 -E '^ *export OMP_NUM_THREADS=' "$1" | sed -E 's/.*=([0-9]+).*/\1/')
  local w=$(( ${np:-1} * ${omp:-1} ))
  [ ${w} -gt ${CORES} ] && w=${CORES}
  echo ${w}
}
if [ ${DRY} = 1 ]; then
  for j in ${JOBS}; do echo "  $(weight ${j#*:})  ${j#*:}"; done
  echo "  ${NJ} 本"; exit 0
fi

# ---------------------------------------------------------------- 実行 (使うコア数の合計を CORES 以下に)
PIDS=(); WS=(); NAMES=(); STARTS=()
USED=0; DONE=0
reap () {   # 終わったジョブを片付ける
  local i n=${#PIDS[@]} np=() nw=() nn=() ns=()
  for ((i = 0; i < n; i++)); do
    if kill -0 ${PIDS[$i]} 2> /dev/null; then
      np+=(${PIDS[$i]}); nw+=(${WS[$i]}); nn+=(${NAMES[$i]}); ns+=(${STARTS[$i]})
    else
      wait ${PIDS[$i]}; local rc=$?
      USED=$(( USED - ${WS[$i]} )); DONE=$(( DONE + 1 ))
      local log=${R}/local_logs/${NAMES[$i]%.sh}.log
      echo "  [done ${DONE}/${NJ}] $(basename ${NAMES[$i]} .sh) rc=${rc} $(( $(date +%s) - ${STARTS[$i]} )) 秒" \
           "$(grep -hoE '^  \[[A-Z_0-9]+\]' ${log} 2> /dev/null | sort | uniq -c | awk '{printf "%s%s ", $2, $1}')"
    fi
  done
  PIDS=(); WS=(); NAMES=(); STARTS=()
  for ((i = 0; i < ${#np[@]}; i++)); do
    PIDS+=(${np[$i]}); WS+=(${nw[$i]}); NAMES+=(${nn[$i]}); STARTS+=(${ns[$i]})
  done
}
for j in ${JOBS}; do
  D=${j%%:*}; f=${j#*:}
  w=$(weight $f)
  while [ $(( USED + w )) -gt ${CORES} ] && [ ${#PIDS[@]} -gt 0 ]; do
    sleep 2; reap
  done
  mkdir -p ${R}/local_logs/${D}
  bash $f > ${R}/local_logs/${D}/$(basename $f .sh).log 2>&1 < /dev/null &
  PIDS+=($!); WS+=(${w}); NAMES+=(${D}/$(basename $f)); STARTS+=($(date +%s))
  USED=$(( USED + w ))
done
while [ ${#PIDS[@]} -gt 0 ]; do sleep 2; reap; done
echo "=== run_local 実行終わり $(date '+%F %T') ($(( $(date +%s) - T_ALL )) 秒、${NJ} 本)"
echo
bash ${R}/summarize.sh --lang "${LSEL}" ${PLATS}
