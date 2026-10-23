#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# 回帰試験の結果 (regression_results/<plat>/<精度>/<ケース>_np<N>/status.txt) を数える。
#
#   使い方 (パッケージのルートで): bash tests/regression/summarize.sh [--mpi yes|no|both] [--lang cpp,c,fortran,python] [<plat> ...]
#     環境を指定しなければ regression_results/ の下にあるものすべて。
#     --mpi  yes = MPI あり (<plat>)、no = MPI なし (<plat>_nompi)、both = 両方。環境を与えたときは
#            MPI ありの名前から <plat>_nompi も選ぶ (既定は与えた名前そのまま、環境を与えなければ全部)。
#     --lang 数える言語。cpp (検証サンプル。lang_* 以外の精度すべて) / c / fortran (f) / python (py)。既定は全部。
#   環境・精度ごとに status の件数を出し、言語別 (C++ / C / Fortran / Python) の合計を出し、最後に OK 以外のケースを並べる。
#   C / Fortran / Python の API の試験は精度の欄が lang_c / lang_f / lang_py で、1 項目 = 1 ケース
#   (1 回の実行で C / Fortran は 12 項目 (先頭が版の version) + 最後の行 all、Python は先頭に has_mpi (GPU の環境は device も) が付く)。
#   終了コードは OK 以外が 1 件でもあれば 1。macOS の /bin/bash (3.2) でも動く書き方にしてある。
set -u
cd "$(dirname "$0")/../.." || exit 1
[ -d regression_results ] || { echo "regression_results/ が無い"; exit 1; }
MPI=""; LSEL="cpp c f py"
ARGS=""
while [ $# -gt 0 ]; do
  case $1 in
    --mpi)    MPI=${2:-}; shift 2 ;;
    --mpi=*)  MPI=${1#--mpi=}; shift ;;
    --lang)   LSEL=${2:-}; shift 2 ;;
    --lang=*) LSEL=${1#--lang=}; shift ;;
    -h|--help) sed -n '3,14p' "$0"; exit 0 ;;
    -*) echo "未知の引数 $1"; exit 2 ;;
    *) ARGS="${ARGS} $1"; shift ;;
  esac
done
case "${MPI}" in ""|yes|no|both) ;; *) echo "--mpi は yes / no / both"; exit 2 ;; esac
# 言語の名前をそろえる (cpp / c / f / py)
LANGS=" "
for L in $(echo "${LSEL}" | tr ',' ' '); do
  case $L in
    cpp|c++|cxx) L=cpp ;; c) ;; f|fortran) L=f ;; py|python) L=py ;;
    *) echo "未知の言語 $L (cpp / c / fortran / python)"; exit 2 ;;
  esac
  LANGS="${LANGS}${L} "
done
# 精度の欄 (ディレクトリ名) -> 言語
lang_of () { case $1 in lang_c) echo c ;; lang_f) echo f ;; lang_py) echo py ;; *) echo cpp ;; esac; }
lang_ok () { case "${LANGS}" in *" $(lang_of "$1") "*) return 0 ;; esac; return 1; }
# 環境の並び
PLATS=""
if [ -z "${ARGS}" ]; then
  for P in $(ls regression_results); do
    case "${MPI}:${P}" in
      yes:*_nompi) continue ;;
      no:*_nompi) ;;
      no:*) continue ;;
    esac
    PLATS="${PLATS} ${P}"
  done
else
  for P in ${ARGS}; do
    B=${P%_nompi}
    case "${MPI}" in
      "")   PLATS="${PLATS} ${P}" ;;
      yes)  PLATS="${PLATS} ${B}" ;;
      no)   PLATS="${PLATS} ${B}_nompi" ;;
      both) PLATS="${PLATS} ${B} ${B}_nompi" ;;
    esac
  done
fi
[ -n "${PLATS}" ] || { echo "数える環境が無い"; exit 1; }
NG=0
for P in ${PLATS}; do
  [ -d regression_results/${P} ] || { echo "${P}: regression_results/${P} が無い"; continue; }
  for d in regression_results/${P}/*/; do
    [ -d "$d" ] || continue
    prec=$(basename "$d")
    lang_ok "${prec}" || continue
    counts=$(cat ${d}*/status.txt 2>/dev/null | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')
    printf "%-10s %-16s %s\n" "${P}" "${prec}" "${counts}"
  done
done
echo
echo "言語別 (lang_c / lang_f / lang_py 以外はすべて C++):"
for P in ${PLATS}; do
  for f in regression_results/${P}/*/*/status.txt; do
    [ -f "$f" ] || continue
    prec=$(basename "$(dirname "$(dirname "$f")")")
    lang_ok "${prec}" || continue
    case ${prec} in
      lang_c) l=C ;; lang_f) l=Fortran ;; lang_py) l=Python ;; *) l=C++ ;;
    esac
    echo "${l} $(cat "$f")"
  done
done | awk '{n[$1]++; if ($2 == "OK") ok[$1]++}
  END {for (l in n) printf "  %-8s ケース %6d  OK %6d  OK 以外 %5d\n", l, n[l], ok[l] + 0, n[l] - ok[l]}' | sort
echo
echo "OK 以外のケース:"
for P in ${PLATS}; do
  for f in regression_results/${P}/*/*/status.txt; do
    [ -f "$f" ] || continue
    lang_ok "$(basename "$(dirname "$(dirname "$f")")")" || continue
    s=$(cat "$f")
    if [ "$s" != OK ]; then
      echo "  ${s}  $(dirname "$f")"
      NG=$((NG+1))
    fi
  done
done
[ ${NG} -eq 0 ] && echo "  (なし)"
exit $(( NG > 0 ))
