# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# s86: setup.sh から読む。利用者のプログラムは spack と同じ GCC 9.5.0 でコンパイルする
#   (Fortran のモジュール parcel_sycl_mod.mod は gfortran の版が違うと読めない)
. /etc/profile.d/modules.sh > /dev/null 2>&1
module load gnu/9.5.0 > /dev/null 2>&1
# Open MPI は spack.yaml で外部登録した 4.1.6 (gnu950, noncuda) と同じものをモジュールで読む。
#   モジュールが PBS (/opt/pbs/lib の libpbs) と UCX のライブラリの場所を足す。無いと mpirun が
#   PBS からノードの一覧を取れず、複数ノードのジョブが 1 ノードに全ランクを載せようとして起動できない。
#   ログインシェルの既定で読み込まれる SGI MPT (mpt) とは衝突するので外してから読む
module unload mpt > /dev/null 2>&1
module load openmpi/cur > /dev/null 2>&1
