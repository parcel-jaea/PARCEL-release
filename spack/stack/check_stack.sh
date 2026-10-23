#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# install_stack.sh で入れた PARCEL の確認 (ログインノードで実行できる範囲)。
#   入っている構成ごとに spack load し, Python から読めるか, C / Fortran / C++ のサンプルがビルドできて
#   実行に要るライブラリがすべて見つかるかを調べる。サンプルの実行ファイルは <出力先>/<構成>/ に置く。
#
#   使い方: bash spack/stack/check_stack.sh <入れ先> <出力先>
set -o pipefail
PREFIX=${1:?入れ先 (install_stack.sh に渡したもの)}
OUT=${2:?出力先}
HERE=$(cd "$(dirname "$0")" && pwd)
EX=${EXAMPLES:-$HERE/../../examples}   # サンプルの場所 (既定は配布パッケージの examples/)
. "$PREFIX/setup.sh"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ng=0
for cfg in $(spack find --format '{variants.mpi}{variants.cuda}{variants.rocm}' parcel-sycl | sort -u); do
  name=$(echo "$cfg" | tr '+~' 'PM')
  echo "=== $cfg"
  (
    spack load "parcel-sycl$cfg" "py-parcel-sycl$cfg" || exit 1
    P=$(spack location -i "parcel-sycl$cfg")
    tr '\n' ' ' < "$P/share/parcel/build_config.txt"; echo
    # MPI ありの版は MPI を初期化する。mpiexec を通さないと初期化できない MPI (富士通 MPI) では MPI_LAUNCH="mpiexec -n 1" を与える
    L_=""; [ "$PARCEL_MPI" = 1 ] && L_=${MPI_LAUNCH:-}
    $L_ python -c "import parcel, numpy; print('python', parcel.build_info(), 'has_mpi', parcel.has_mpi())" || exit 1
    # MPI なしの Fortran は sample_f.f90 の "use mpi" を 1 プロセス用の代用モジュールで満たす
    #   コンパイラの名前は環境変数で変えられる (富士通: CHECK_CC=fcc CHECK_FC=frt MPICC=mpifcc MPIFC=mpifrt MPICXX=mpiFCC)。
    #   CC / FC も見るが spack load (LLVM) が CC を clang に書き換えるので, CHECK_CC / CHECK_FC を優先する。
    #   追加のフラグは CFLAGS / FFLAGS (富士通の frt は -Kopenmp が要る。envs/wis_ody/after_activate.sh)。
    #   リンクで先に探させる場所は CHECK_LDFLAGS (spack で作った GCC の実行時ライブラリなど)
    if [ "$PARCEL_MPI" = 1 ]; then CC_=${MPICC:-mpicc}; FC_=${MPIFC:-mpif90}; CXX_=${MPICXX:-mpicxx}; STUB=
    else CC_=${CHECK_CC:-${CC:-gcc}}; FC_=${CHECK_FC:-${FC:-gfortran}}; CXX_=acpp; STUB=$EX/sample_f_mpi_stub.f90; fi
    D=$OUT/$name; mkdir -p "$D"; cd "$D" || exit 1
    $CC_ ${CFLAGS:-} ${CHECK_LDFLAGS:-} -O2 -I"$P/include" -o sample_c "$EX/sample_c.c" -L"$P/lib" -lparcel_sycl_c -Wl,-rpath,"$P/lib" -lm > build_c.log 2>&1 \
      || { echo "sample_c build failed"; tail -5 build_c.log; exit 1; }
    $FC_ ${FFLAGS:-} ${CHECK_LDFLAGS:-} -O2 -I"$P/include" -o sample_f $STUB "$EX/sample_f.f90" -L"$P/lib" -lparcel_sycl_f -lparcel_sycl_c -Wl,-rpath,"$P/lib" > build_f.log 2>&1 \
      || { echo "sample_f build failed"; tail -5 build_f.log; exit 1; }
    $CXX_ ${CHECK_LDFLAGS:-} $PARCEL_CXXFLAGS -O3 -o quick_start "$EX/quick_start.cpp" $PARCEL_LDFLAGS > build_cpp.log 2>&1 \
      || { echo "quick_start build failed"; tail -5 build_cpp.log; exit 1; }
    for f in sample_c sample_f quick_start; do
      if command -v ldd > /dev/null; then n=$(ldd "$f" | grep -c 'not found')
      else n=$(otool -L "$f" | awk 'NR>1 && $1 ~ /^\// {print $1}' | while read l; do [ -e "$l" ] || echo "$l"; done | wc -l | tr -d ' ')   # macOS
      fi
      echo "  $f built, libraries not found: $n"; [ "$n" = 0 ] || exit 1
    done
  ) && echo "OK $cfg" || { echo "NG $cfg"; ng=$((ng+1)); }
done
echo "=== check_stack: NG=$ng"
exit $ng
