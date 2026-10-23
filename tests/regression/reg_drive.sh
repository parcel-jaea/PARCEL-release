#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# feed_regression.sh を繰り返し呼び、試験ジョブを小分けに投入し続ける。
#   済んだケースは試験ジョブが飛ばすので、再投入しても残りだけが走る。
#
#   使い方 (パッケージのルートで、切り離して起動):
#     nohup bash tests/regression/reg_drive.sh <plat> <同時上限> [待つビルドジョブ ID] [種別] &
#       種別は feed_regression.sh と同じ (regression / cpp / lang / lang_c / lang_f / lang_py / dd / pspec / perf / prec。既定 regression)
#     5 番目以降に --mpi yes|no|both / --lang cpp,c,fortran,python を与えると feed_regression.sh にそのまま渡す
#       (例: nohup bash tests/regression/reg_drive.sh s86_cpu 20 "" sel --mpi both --lang c,fortran &)
#   - 待つビルドジョブ ID を与えると、そのジョブがキューから消えるまで投入しない。
#   - 全ジョブの投入が済み、キューからも消えたら "REG_DRIVE_DONE" を出して終わる。
#   ログ: tests/regression/reg_drive_<種別>_<plat>.log
set -u
cd "$(dirname "$0")/../.." || exit 1
P=$1; CAP=$2; WAITJOB=${3:-}; KIND=${4:-regression}
LOG=tests/regression/reg_drive_${KIND}_${P}.log
if command -v pjstat >/dev/null 2>&1; then QLIST="pjstat"; else
  case "$(hostname)" in miyabi*) QLIST="qstat" ;; *) QLIST="qstat -u ${USER:-$(id -un)}" ;; esac
  QLIST=${PARCEL_QLIST:-${QLIST}}
fi
exec >> ${LOG} 2>&1
echo "=== $(date '+%F %T') start ${P} cap=${CAP} waitjob=${WAITJOB}"
if [ -n "${WAITJOB}" ]; then
  while true; do
    Q=$(${QLIST} 2>&1) || { sleep 60; continue; }      # 取得失敗は「消えた」と見なさない
    grep -q "^${WAITJOB}" <<< "${Q}" || break
    sleep 60
  done
  echo "  $(date '+%T') build job ${WAITJOB} finished"
fi
while true; do
  OUT=$(bash tests/regression/feed_regression.sh ${P} ${CAP} ${KIND} "${@:5}" 2>&1)
  L=$(tail -1 <<< "${OUT}")
  echo "  $(date '+%T') ${L}"
  grep -q "投入できず" <<< "${OUT}" && grep "投入できず" <<< "${OUT}"
  left=$(grep -oE 'left=[0-9]+' <<< "${L}" | cut -d= -f2)
  active=$(grep -oE 'active=[0-9]+' <<< "${L}" | cut -d= -f2)
  total=$(grep -oE 'total=[0-9]+' <<< "${L}" | cut -d= -f2)
  if [ "${left:-x}" = 0 ] && [ "${active:-x}" = 0 ] && [ "${total:-0}" -gt 0 ]; then
    echo "=== $(date '+%F %T') REG_DRIVE_DONE ${P}"; break
  fi
  sleep 90
done
