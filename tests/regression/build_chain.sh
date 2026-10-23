#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# 実行ファイルのビルド (sub_build_<plat>.sh) を、指定した環境の順に 1 つずつ行う (ログインノードで実行)。
#   同じログインノードで CPU 用と GPU 用のビルドを同時に走らせないためのもの
#   (並列ビルドを 2 つ重ねるとログインノードのコアとメモリを倍使う)。
#   各環境の結果を tests/regression/build_chain_<plat>.done に書く (実行ファイルの本数)。
#
#   使い方 (パッケージのルートで、切り離して起動):
#     setsid nohup bash tests/regression/build_chain.sh s86_cpu s86_gpu > tests/regression/build_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.." || exit 1
for p in "$@"; do
  rm -f tests/regression/build_chain_${p}.done
  echo "=== $(date '+%F %T') build ${p} 開始 ==="
  T0=$(date +%s)
  bash tests/regression/sub_build_${p}.sh > tests/regression/build_chain_${p}.log 2>&1 < /dev/null
  RC=$?
  n=$(ls bin/${p}/verify_* 2>/dev/null | wc -l)
  echo "=== $(date '+%F %T') build ${p} 終了 rc=${RC} 実行ファイル ${n} 本 ($(( $(date +%s)-T0 )) 秒) ==="
  grep -E '\[OK\]|\[FAIL\]' tests/regression/build_chain_${p}.log | sed 's/^/  /'
  echo "${n}" > tests/regression/build_chain_${p}.done
done
echo "=== $(date '+%F %T') ALL DONE ==="
