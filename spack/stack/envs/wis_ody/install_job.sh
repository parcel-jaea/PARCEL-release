#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Wisteria-Odyssey で install_stack.sh を計算ノードで流すジョブの例 (ログインノードは x86 なので A64FX 向けは計算ノードで作る)
#   使い方: パッケージのトップで  pjsub -g <グループ> -x PREFIX=<入れ先> spack/stack/envs/wis_ody/install_job.sh
#   時間: LLVM 18 のビルドに -j12 で約 3 時間、依存パッケージ全体で約 4 時間 30 分 (実測)。
#         上限で止まっても、同じジョブを流し直せば入ったものは使い回される。
#   並列数: envs/wis_ody/site_env.sh の PARCEL_STACK_JOBS (既定 12)。32 以上はメモリ不足になる。
#PJM -L "rscgrp=small-o"
#PJM -L "node=1"
#PJM -L "elapse=24:00:00"
#PJM -j
cd "${PJM_O_WORKDIR:-$(pwd)}" || exit 1
: "${PREFIX:?PREFIX=<入れ先> を -x で与える}"
echo "=== $(date) $(hostname) install_stack wis_ody -> ${PREFIX}"
bash spack/stack/install_stack.sh wis_ody "${PREFIX}"
rc=$?
echo "=== $(date) END rc=${rc}"
exit ${rc}
