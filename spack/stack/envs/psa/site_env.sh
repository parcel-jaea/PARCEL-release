# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# PS-A (x86_64 Xeon 6980P): setup.sh と install_stack.sh から読む。サイトの MPI などの module
. /etc/profile.d/modules.sh > /dev/null 2>&1
module purge > /dev/null 2>&1
module load intel intelmpi > /dev/null 2>&1
