# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# macOS (Apple Silicon) + Homebrew の環境設定 (source して使う)。ビルド・pip install・実行で同じ設定にする。
#   前提: brew install adaptivecpp open-mpi openblas gcc numpy scipy
#   Homebrew の AdaptiveCpp は LLVM を組み込まない構成で, カーネルを含むコード全体をホストの clang++ でコンパイルする
#   (library-only。--acpp-targets=omp と omp.library-only は同じコンパイル行になる)。他の CPU と同じく縮約だけを
#   ブロック化する (-D_BLOCKING_LOOP_DOT_=1)。1 行 = 1 work item の平坦なループのままだと内積などが極端に遅い。
#   全カーネルのブロック化 (-D_BLOCKING_LOOP_=1、A64FX 向け) との差は Apple M4 で 1 反復 ±4 % 以内で、
#   こちらは Jacobi 前処理の値を保持する (メモリが増える) ので使わない。
#     PARCEL_BREW   Homebrew の場所 (既定 /opt/homebrew)
#     PARCEL_MPI    既定 0 = MPI なし (acpp でビルドする)。1 で MPI あり (Open MPI の mpicxx, 中で acpp を呼ぶ)
_brew=${PARCEL_BREW:-/opt/homebrew}
export PATH=${_brew}/bin:${PATH}
export PARCEL_MPI=${PARCEL_MPI:-0}
export OMPI_CXX=acpp MPICXX_CXX=acpp
# Python は Homebrew の numpy / scipy と同じ版 (venv はこれで作る)
export PARCEL_PYTHON=${_brew}/bin/python3.14
# ネイティブ層のビルド設定
export PARCEL_TARGET=omp.library-only
export PARCEL_KERNEL_FLAGS="-D_BLOCKING_LOOP_DOT_=1"
export PARCEL_HOST_FLAGS=""
export PARCEL_EXTRA_FLAGS="-Wl,-rpath,${_brew}/opt/openblas/lib"
export LAPACKE_INCLUDE_DIR=${_brew}/opt/openblas/include
export LAPACK_LIBS="-L${_brew}/opt/openblas/lib -lopenblas"
# examples/build_examples.sh (C++ / C / Fortran のサンプル) に渡す値
export SYCL_TARGET="--acpp-targets=${PARCEL_TARGET}"
export CXXFLAGS="-O3 ${PARCEL_KERNEL_FLAGS}"
export LAPACK_INC="-I${LAPACKE_INCLUDE_DIR}"
export LAPACK_LIB="${LAPACK_LIBS} -Wl,-rpath,${_brew}/opt/openblas/lib"
# gfortran の実行時ライブラリ (macOS は .dylib なので build_examples.sh の既定の探し方が使えない)
export FC_LINK="-L${_brew}/opt/gcc/lib/gcc/current -lgfortran -Wl,-rpath,${_brew}/opt/gcc/lib/gcc/current"
[ "${PARCEL_MPI}" = 1 ] && export FC_LINK="$(mpif90 --showme:link) ${FC_LINK}"
[ "${PARCEL_MPI}" = 0 ] && export MPICC=clang MPIFC=gfortran
unset _brew
