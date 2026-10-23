#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# 回帰試験の結果から反復数を抜き出して CSV で標準出力に出す (各ホストで実行する)。
#
#   使い方 (パッケージのルートで): bash tests/regression/collect_iters.sh <plat> [<plat> ...] > iters_<host>.csv
#   集計は tests/regression/compare_iters.py に渡す。
#   MPI なしの構成は <plat>_nompi を指定する。MPI あり np=1 との突き合わせは compare_nompi.py (この CSV も読める)。
#
#   列: plat,prec,case,np,status,iters,outer,date
#     iters : 反復履歴の最後の "N : residual norm" の N (線形ソルバ・内部解法)
#     outer : "iterations = N" / "restarts = N" (inverse_power / power / ks_arnoldi の外側)
#     date  : status.txt の更新日 (環境ごとに回した日が違うので突き合わせのとき確認する)
#   合否 (status) はそのまま写すだけで、ここでは判定しない。
set -u
cd "$(dirname "$0")/../.." || exit 1
echo "plat,prec,case,np,status,iters,outer,date"
for P in "$@"; do
  for f in regression_results/${P}/*/*/status.txt; do
    [ -f "$f" ] || continue
    d=$(dirname "$f"); prec=$(basename "$(dirname "$d")"); c=$(basename "$d")
    base=${c%_np*}; np=${c##*_np}
    R=$d/parcel_result.txt; [ -s "$R" ] || R=$d/result.txt
    it=$(grep -E "^ *[0-9]+ : residual norm" "$R" 2>/dev/null | tail -1 | awk '{print $1}')
    outer=$(grep -E "(iterations|restarts) *= *[0-9]+" "$R" 2>/dev/null | tail -1 | grep -oE "= *[0-9]+" | tr -dc 0-9)
    echo "${P},${prec},${base},${np},$(cat "$f"),${it},${outer},$(stat -c %y "$f" | cut -c1-10)"
  done
done
