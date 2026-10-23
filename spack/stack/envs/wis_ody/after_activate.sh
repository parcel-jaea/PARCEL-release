# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Wisteria-Odyssey: 利用者のプログラムは富士通のコンパイラと MPI ラッパで組む (+fj の Fortran のモジュールは frt で作ってある)
export CC=fcc FC=frt MPICC=mpifcc MPIFC=mpifrt MPICXX=mpiFCC
export CHECK_CC=fcc CHECK_FC=frt   # check_stack.sh 用 (spack load parcel-sycl は LLVM の設定で CC を clang に書き換える)
export MPI_LAUNCH="mpiexec -n 1"   # 富士通 MPI は mpiexec を通さないと MPI を初期化できない (check_stack.sh が使う)
# frt で -Kopenmp を付けずに組んだプログラムから PARCEL (富士通の OpenMP で並列化) を複数スレッドで呼ぶと,
#   線形ソルバの結果が壊れる (sample_f で確認。-Kopenmp 付き 12 スレッド・付けない 1 スレッドは合格)。C も念のため付ける
export FFLAGS=-Kopenmp CFLAGS=-Kopenmp
