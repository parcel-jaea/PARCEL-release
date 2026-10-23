# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# PS-B (MI300A, 計算ノードは Zen 4 / ログインノードは Xeon): setup.sh と install_stack.sh から読む。サイトの MPI などの module
. /etc/profile.d/modules.sh > /dev/null 2>&1
module purge > /dev/null 2>&1
# rocm/6.3.3 も読まれ ROCM_PATH が設定される (AdaptiveCpp の ROCm 版のビルドが使う)
module load openmpi/5.0.7/rocm6.3.3 > /dev/null 2>&1
