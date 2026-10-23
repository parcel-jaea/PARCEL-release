# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Wisteria-Odyssey: setup.sh と install_stack.sh から読む。富士通のコンパイラと MPI の module (A64FX の計算ノードで読む)
. /etc/profile.d/modules.sh > /dev/null 2>&1
module purge > /dev/null 2>&1
module load odyssey > /dev/null 2>&1
# 同時コンパイル数: A64FX の 1 ノードは HBM2 32 GiB (利用者のジョブは約 28 GiB)。-j32 では LLVM のビルドで
#   メモリが足りずに止まった (ピーク約 29 GiB)。-j12 で LLVM 18 のビルドに約 3 時間、依存パッケージ全体で約 4 時間 30 分
export PARCEL_STACK_JOBS=${PARCEL_STACK_JOBS:-12}
