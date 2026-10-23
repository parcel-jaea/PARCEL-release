# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Python パッケージ parcel のネイティブ層 (libparcel_py.so) のビルド。
#   pip install . から scikit-build-core 経由で呼ばれる。設定は -C cmake.define.<名前>=<値>
#   (または環境変数 CMAKE_ARGS="-D<名前>=<値> ...") で与える。
#
#   PARCEL_TARGET       AdaptiveCpp のターゲット (omp, cuda:sm_70, cuda:sm_80, cuda:sm_90, hip:gfx942 など)。既定 omp
#   PARCEL_KERNEL_FLAGS カーネル最適化のマクロ。既定は PARCEL_TARGET から決める
#                       (omp: -D_BLOCKING_LOOP_DOT_=1, hip: -DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1,
#                        cuda: 梯子分割なし (-DPARCEL_SPMV_LADDER=0 -DPARCEL_PRECON_LADDER=0))。
#                       A64FX (omp.library-only) は推奨オプションに合わせて明示すること
#   PARCEL_HOST_FLAGS   ホスト向けの最適化。既定 -march=native -mtune=native
#                       (A64FX は -ffp-contract=fast。富士通コンパイラは既定で積和を FMA にまとめない)
#   PARCEL_EXTRA_FLAGS  そのほかのフラグ (既定なし)
#   PARCEL_INCLUDE_DIR  PARCEL の Include/。既定はこのディレクトリの ../Include
#   LAPACKE_INCLUDE_DIR lapacke.h の場所。既定 /usr/include/openblas
#   LAPACK_LIBS         LAPACK のライブラリ。既定 -lopenblas
#   PARCEL_MPI          0: MPI なしの 1 プロセス版 (既定。-DPARCEL_MPI=0。MPI のライブラリをリンクしない), 1: MPI あり
#   PARCEL_MPICXX       C++ コンパイラ。既定は MPI ありなら mpicxx (中で acpp を呼ぶよう OMPI_CXX=acpp などを
#                       設定しておく), MPI なしなら acpp
#
#   コンパイラは CMake の言語機能ではなく MPI ラッパを直接呼ぶ (AdaptiveCpp と MPI ラッパの組合せを
#   CMake のコンパイラ判定に通さないため)。環境 (module, spack, LD_LIBRARY_PATH) はビルド前に整えておく。
cmake_minimum_required(VERSION 3.18)
project(parcel_py LANGUAGES NONE)

set(PARCEL_TARGET "omp" CACHE STRING "AdaptiveCpp target (--acpp-targets=...)")
set(PARCEL_KERNEL_FLAGS "" CACHE STRING "kernel optimization macros (empty: chosen from PARCEL_TARGET)")
set(PARCEL_HOST_FLAGS "-march=native -mtune=native" CACHE STRING "host optimization flags")
set(PARCEL_EXTRA_FLAGS "" CACHE STRING "extra compiler flags")
set(PARCEL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Include" CACHE PATH "PARCEL Include/ directory")
set(LAPACKE_INCLUDE_DIR "/usr/include/openblas" CACHE PATH "directory of lapacke.h")
set(LAPACK_LIBS "-lopenblas" CACHE STRING "LAPACK libraries")
set(PARCEL_MPI "0" CACHE STRING "0: without MPI (single process, default), 1: with MPI")
set(PARCEL_MPICXX "" CACHE STRING "C++ compiler (empty: mpicxx with MPI, acpp without MPI)")

if(NOT EXISTS "${PARCEL_INCLUDE_DIR}/parcel_sycl.hpp")
  message(FATAL_ERROR "parcel_sycl.hpp not found in PARCEL_INCLUDE_DIR=${PARCEL_INCLUDE_DIR}")
endif()
if(PARCEL_KERNEL_FLAGS STREQUAL "")
  if(PARCEL_TARGET MATCHES "^cuda")
    # NVIDIA の GPU: 梯子分割は付けない (V100 は平坦より遅く、A100・H100 は差が無かった。明示の 0 は既定と同じ平坦)
    set(PARCEL_KERNEL_FLAGS "-DPARCEL_SPMV_LADDER=0 -DPARCEL_PRECON_LADDER=0")
  elseif(PARCEL_TARGET MATCHES "^hip")
    set(PARCEL_KERNEL_FLAGS "-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1")
  else()
    set(PARCEL_KERNEL_FLAGS "-D_BLOCKING_LOOP_DOT_=1")
  endif()
endif()

if(PARCEL_MPI STREQUAL "0")
  set(_mpi_def "-DPARCEL_MPI=0")
  if(PARCEL_MPICXX STREQUAL "")
    set(PARCEL_MPICXX "acpp")
  endif()
elseif(PARCEL_MPI STREQUAL "1")
  set(_mpi_def "")
  if(PARCEL_MPICXX STREQUAL "")
    set(PARCEL_MPICXX "mpicxx")
  endif()
else()
  message(FATAL_ERROR "PARCEL_MPI must be 0 or 1 (got ${PARCEL_MPI})")
endif()

separate_arguments(_kernel UNIX_COMMAND "${PARCEL_KERNEL_FLAGS}")
separate_arguments(_host   UNIX_COMMAND "${PARCEL_HOST_FLAGS}")
separate_arguments(_extra  UNIX_COMMAND "${PARCEL_EXTRA_FLAGS}")
separate_arguments(_lapack UNIX_COMMAND "${LAPACK_LIBS}")

set(_out "${CMAKE_CURRENT_BINARY_DIR}/libparcel_py.so")
set(_cmd ${PARCEL_MPICXX} -O3 --acpp-targets=${PARCEL_TARGET} ${_mpi_def} ${_host} ${_kernel} ${_extra}
         -fPIC -shared -I${PARCEL_INCLUDE_DIR} -I${LAPACKE_INCLUDE_DIR}
         -o "${_out}" "${CMAKE_CURRENT_SOURCE_DIR}/src/parcel_py.cpp" ${_lapack})
# コンパイルの指定をファイルに残し, 変わったときだけ書き換える (file(CONFIGURE) は内容が同じなら触らない)。
#   これを依存に加え, 同じビルドディレクトリで構成 (CPU / GPU, MPI なし / あり) を替えたとき
#   前の構成のライブラリが作り直されずに使われるのを防ぐ
string(REPLACE ";" " " _cmd_str "${_cmd}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build_command.txt" CONTENT "${_cmd_str}\n")
add_custom_command(
  OUTPUT "${_out}"
  COMMAND ${_cmd}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/parcel_py.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/parcel_py.h"
          "${CMAKE_CURRENT_BINARY_DIR}/build_command.txt"
  COMMENT "Building libparcel_py.so (--acpp-targets=${PARCEL_TARGET} ${PARCEL_KERNEL_FLAGS} PARCEL_MPI=${PARCEL_MPI})"
  VERBATIM)
add_custom_target(parcel_py ALL DEPENDS "${_out}")

# ビルドの設定をパッケージに残す (parcel.build_info() で表示する)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py"
  "TARGET = \"${PARCEL_TARGET}\"\nKERNEL_FLAGS = \"${PARCEL_KERNEL_FLAGS}\"\nHOST_FLAGS = \"${PARCEL_HOST_FLAGS}\"\nMPI = ${PARCEL_MPI}\n")

install(FILES "${_out}" "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py" DESTINATION parcel)
