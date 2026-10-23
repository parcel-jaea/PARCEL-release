#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# 試験ジョブを、同時に投入しておく本数の上限を守りながら 1 回ぶん投入する (ループから定期的に呼ぶ)。
#
#   使い方 (パッケージのルートで):
#     bash tests/regression/feed_regression.sh <plat> <上限> [regression|dd|pspec|lang|perf|prec] [接尾辞]
#       regression : tests/regression/sub_regression_<plat>_*.sh (既定)
#       dd         : calcDD / dataDD のジョブだけ
#       pspec      : A M^-1 固有値モードのジョブ (sub_regression_<plat>_*_pspec_np*.sh) だけ
#       lang       : C / Fortran / Python の API の試験 (sub_regression_<plat>_lang_*_np*.sh) だけ。
#                    sub_build_lang_<plat>.sh が書く bin/<plat>/lang_built.txt を待つ
#       cpp        : 検証サンプル (全精度) と ddcheck のジョブだけ (言語別 _lang_ を除く)
#       lang_c / lang_f / lang_py : その言語のジョブだけ。bin/<plat>/lang_built.txt にその言語の行があるのを待つ
#     bash tests/regression/feed_regression.sh <plat> <上限> [--mpi yes|no|both] [--lang cpp,c,fortran,python]
#       選んだ組 (<plat> と <plat>_nompi、言語ごとの種別 cpp / lang_c / lang_f / lang_py) を順に 1 回ずつ流す。上限は全体で数える
#       (前の組のキューに残っている本数を引いて次の組に渡す)。--mpi を省くと <plat> そのまま、--lang を省くと全言語。
#       最後の行は FEEDSEL <plat> ... left=<未投入> active=<キュー中> total=<投入済み> (合計)。
#     regression (既定) は cpp と言語別の両方を含む。同じジョブを種別を変えて二重に投入しないこと
#     (投入済みの記録は種別ごとに別のファイル)。
#       perf       : 性能試験 tests/perf/sub_perf_*_<plat><接尾辞>.sh (接尾辞は _nx512 など)
#       prec       : 混合精度の性能試験 tests/perf/sub_perf_*_<plat>_prec_*.sh
#
#   - 実行ファイル bin/<plat>/verify_* が精度の数だけそろうまで投入しない (perf は verify_double、prec は
#     tests/perf/sub_build_prec_<plat>.sh が作る bin/<plat>/perf_prec/verify_*)。
#   - MPI なしの構成は <plat>_nompi を指定する (bin/<plat>_nompi/、sub_regression_<plat>_nompi_*.sh)。
#   - 投入したジョブ ID を tests/regression/feed_<種別>_<plat>.jobs に記録し、キューに残っている本数で
#     上限を判定する (qstat / pjstat はジョブ名を切り詰めるので名前では数えない)。
#     まだ投入していないスクリプトの一覧は同じ名前の .state に置く。やり直すときは両方を消す。
#   - ジョブ ID が返らなければその回は打ち切る (投入数の上限に当たったときなど)。
#   出力の最終行: FEED <plat> bins=<n>/<必要数> left=<未投入> active=<キュー中> sent=<今回> total=<投入済み>
set -u
cd "$(dirname "$0")/../.." || exit 1
R=tests/regression
# --mpi / --lang: 選んだ組を順に流す
SEL_MPI=""; SEL_LANG=""; POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --mpi)    SEL_MPI=$2; shift 2 ;;
    --mpi=*)  SEL_MPI=${1#--mpi=}; shift ;;
    --lang)   SEL_LANG=$2; shift 2 ;;
    --lang=*) SEL_LANG=${1#--lang=}; shift ;;
    *) POS+=("$1"); shift ;;
  esac
done
set -- "${POS[@]}"
if [ -n "${SEL_MPI}${SEL_LANG}" ]; then
  P=$1; CAP=$2; B=${P%_nompi}
  case "${SEL_MPI}" in
    "")   PL="${P}" ;;
    yes)  PL="${B}" ;;
    no)   PL="${B}_nompi" ;;
    both) PL="${B} ${B}_nompi" ;;
    *) echo "--mpi は yes / no / both"; exit 2 ;;
  esac
  KINDS=""
  for L in $(echo "${SEL_LANG:-cpp,c,f,py}" | tr ',' ' '); do
    case $L in
      cpp|c++|cxx) KINDS="${KINDS} cpp" ;; c) KINDS="${KINDS} lang_c" ;;
      f|fortran) KINDS="${KINDS} lang_f" ;; py|python) KINDS="${KINDS} lang_py" ;;
      *) echo "未知の言語 $L (cpp / c / fortran / python)"; exit 2 ;;
    esac
  done
  REM=${CAP}; LEFT=0; ACT=0; TOT=0
  for p in ${PL}; do
    for k in ${KINDS}; do
      case $k in
        cpp) ls ${R}/sub_regression_${p}_*.sh 2> /dev/null | grep -v "_${p}_nompi_" | grep -qv "_${p}_lang_" || continue ;;
        *)   ls ${R}/sub_regression_${p}_${k}_np*.sh > /dev/null 2>&1 || continue ;;
      esac
      [ ${REM} -lt 0 ] && REM=0
      OUT=$(bash ${R}/feed_regression.sh "${p}" "${REM}" "${k}")
      echo "${OUT}"
      L1=$(tail -1 <<< "${OUT}")
      a=$(grep -oE 'active=[0-9]+' <<< "${L1}" | cut -d= -f2); l=$(grep -oE 'left=[0-9]+' <<< "${L1}" | cut -d= -f2)
      t=$(grep -oE 'total=[0-9]+' <<< "${L1}" | cut -d= -f2)
      REM=$((REM - ${a:-0})); ACT=$((ACT + ${a:-0})); LEFT=$((LEFT + ${l:-0})); TOT=$((TOT + ${t:-0}))
    done
  done
  echo "FEEDSEL ${P} mpi=${SEL_MPI:-asis} lang=${SEL_LANG:-all} left=${LEFT} active=${ACT} total=${TOT}"
  exit 0
fi
P=$1; CAP=$2; KIND=${3:-regression}; SFX=${4:-}
if [ "${KIND}" = prec ]; then
  ST=${R}/feed_prec_${P}.state; JOBS=${R}/feed_prec_${P}.jobs; TAG=FEEDPREC; NEED=1
  LIST="ls tests/perf/sub_perf_*_${P}_prec_*.sh"; BINS="ls bin/${P}/perf_prec/verify_*"
elif [ "${KIND}" = perf ]; then
  ST=${R}/feed_perf${SFX}_${P}.state; JOBS=${R}/feed_perf${SFX}_${P}.jobs; TAG=FEEDPERF${SFX}; NEED=1
  LIST="ls tests/perf/sub_perf_*_${P}${SFX}.sh"; BINS="ls bin/${P}/verify_double"
else
  # NEED は精度構成の数。gen_regression.py の PRECISIONS から数える
  NEED=$(python3 -c "
import re
s=open('${R}/gen_regression.py').read()
m=re.search(r'^PRECISIONS = \{(.*?)^\}', s, re.M|re.S)
print(len(re.findall(r'^\s{4}\"[A-Za-z0-9_]+\":', m.group(1), re.M)) if m else 0)
" 2>/dev/null)
  case "${NEED}" in ''|*[!0-9]*|0) NEED=13 ;; esac
  if [ "${KIND}" = dd ]; then
    ST=${R}/feed_dd_${P}.state; JOBS=${R}/feed_dd_${P}.jobs; TAG=FEEDDD
    LIST="ls ${R}/sub_regression_${P}_calcDD_*.sh ${R}/sub_regression_${P}_dataDD_*.sh"
  elif [ "${KIND}" = lang ]; then
    ST=${R}/feed_lang_${P}.state; JOBS=${R}/feed_lang_${P}.jobs; TAG=FEEDLANG
    LIST="ls ${R}/sub_regression_${P}_lang_*_np*.sh"
  elif [ "${KIND}" = cpp ]; then
    # 検証サンプルと ddcheck だけ。言語別 (_lang_) は下で除く
    ST=${R}/feed_cpp_${P}.state; JOBS=${R}/feed_cpp_${P}.jobs; TAG=FEEDCPP
    LIST="ls ${R}/sub_regression_${P}_*.sh"
  elif [ "${KIND}" = lang_c ] || [ "${KIND}" = lang_f ] || [ "${KIND}" = lang_py ]; then
    ST=${R}/feed_${KIND}_${P}.state; JOBS=${R}/feed_${KIND}_${P}.jobs; TAG=FEED$(echo ${KIND} | tr a-z A-Z)
    LIST="ls ${R}/sub_regression_${P}_${KIND}_np*.sh"
  elif [ "${KIND}" = pspec ]; then
    ST=${R}/feed_pspec_${P}.state; JOBS=${R}/feed_pspec_${P}.jobs; TAG=FEEDPSPEC
    LIST="ls ${R}/sub_regression_${P}_*_pspec_np*.sh"
  else
    ST=${R}/feed_${P}.state; JOBS=${R}/feed_${P}.jobs; TAG=FEED
    LIST="ls ${R}/sub_regression_${P}_*.sh"
  fi
  BINS="ls bin/${P}/verify_*"
  # C / Fortran / Python は verify_* ではなく、言語別のビルドの完了の印 (1 本) を待つ
  [ "${KIND}" = lang ] && { NEED=1; BINS="ls bin/${P}/lang_built.txt"; }
  case "${KIND}" in lang_*) NEED=1; BINS="grep -x ${KIND#lang_} bin/${P}/lang_built.txt" ;; esac
fi
touch ${JOBS}
if [ ! -f ${ST} ]; then
  # MPI なしの環境 <plat>_nompi のスクリプトは sub_regression_<plat>_*.sh にも一致するので、
  #   MPI ありの環境を投入するときは除く
  case "${P}" in
    *_nompi) ${LIST} > ${ST} ;;
    *)       ${LIST} | grep -v "_${P}_nompi_" > ${ST} ;;
  esac
  if [ "${KIND}" = cpp ]; then
    grep -v "_${P}_lang_" ${ST} > ${ST}.tmp; mv ${ST}.tmp ${ST}
  fi
fi

if command -v pjsub >/dev/null 2>&1; then
  SUBMIT=pjsub; QLIST="pjstat"
else
  SUBMIT=qsub
  # 自分のジョブの一覧を出すコマンド。PARCEL_QLIST で上書きできる
  #   (Miyabi の qstat は自分のジョブだけを表示し、-u を付けない)
  case "$(hostname)" in
    miyabi*) QLIST="qstat" ;;
    *)       QLIST="qstat -u ${USER:-$(id -un)}" ;;
  esac
  QLIST=${PARCEL_QLIST:-${QLIST}}
fi

bins=$(${BINS} 2>/dev/null | wc -l)
# キューに残っているジョブ ID (数字部分)
QIDS=$(${QLIST} 2>/dev/null | grep -oE '^[0-9]+' | sort -u)
active=0
while read -r id; do
  [ -z "${id}" ] && continue
  grep -qx "${id%%.*}" <<< "${QIDS}" && active=$((active+1))
done < ${JOBS}

sent=0
if [ "${bins}" -lt ${NEED} ]; then
  echo "  ${P}: ビルド待ち (${bins}/${NEED})"
else
  while [ -s ${ST} ] && [ ${active} -lt ${CAP} ]; do
    f=$(head -1 ${ST})
    OUT=$(${SUBMIT} "${f}" < /dev/null 2>&1)
    ID=$(grep -oE '^[0-9]+\.[A-Za-z0-9]+|Job [0-9]+ submitted' <<< "${OUT}" | grep -oE '^[0-9]+(\.[A-Za-z0-9]+)?|[0-9]+' | head -1)
    if [ -z "${ID}" ]; then
      echo "  ${P}: 投入できず打ち切り (${f}): $(tr '\n' ' ' <<< "${OUT}" | cut -c1-160)"
      break
    fi
    echo "${ID}" >> ${JOBS}
    sed -i '1d' ${ST}
    sent=$((sent+1)); active=$((active+1))
  done
fi
echo "${TAG} ${P} bins=${bins}/${NEED} left=$(wc -l < ${ST}) active=${active} sent=${sent} total=$(wc -l < ${JOBS})"
