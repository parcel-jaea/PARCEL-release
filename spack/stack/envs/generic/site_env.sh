# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# generic: install_stack.sh と setup.sh から読む (雛形)
#   spack.yaml で外部登録した MPI・CUDA を, インストール中と利用時の両方で使えるようにする.
#   モジュールの無い計算機では, この中身は空のままでよい.
#
# 例: 環境モジュール (module) を使う計算機. モジュールの名前は module avail で調べる
# . /etc/profile.d/modules.sh > /dev/null 2>&1
# module load gcc openmpi cuda > /dev/null 2>&1
#
# 例: RHEL 系の OS の Open MPI (dnf の openmpi-devel) は環境モジュールで読む
# . /etc/profile.d/modules.sh > /dev/null 2>&1
# module load mpi/openmpi-x86_64 > /dev/null 2>&1
#
# 同時コンパイル数 (install_stack.sh の第 3 引数を省いたときに使う. 既定 16).
#   LLVM のビルドは 1 本あたりのメモリが大きいので, メモリが少ない計算機では下げる
# PARCEL_STACK_JOBS=8
