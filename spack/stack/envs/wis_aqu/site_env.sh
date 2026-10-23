# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Wisteria-Aquarius: setup.sh と install_stack.sh から読む。サイトの MPI と CUDA の module
. /etc/profile.d/modules.sh > /dev/null 2>&1
module purge > /dev/null 2>&1
module load aquarius cuda ompi > /dev/null 2>&1
