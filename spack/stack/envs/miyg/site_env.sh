# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Miyabi-G (aarch64 Grace + GH200): setup.sh と install_stack.sh から読む。サイトの MPI などの module
. /etc/profile.d/modules.sh > /dev/null 2>&1
module purge > /dev/null 2>&1
module load gcc cuda ompi-cuda > /dev/null 2>&1
