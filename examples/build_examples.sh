#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# examples/ のサンプルをビルドする (AdaptiveCpp + Open MPI + OpenBLAS の例)。
#
#   使い方 (パッケージのルートで。SYCL 処理系と MPI の環境を整えてから):
#     export OMPI_CXX=acpp            # mpicxx の中身を AdaptiveCpp にする (Open MPI の場合)
#     bash examples/build_examples.sh            # 全部
#     bash examples/build_examples.sh quick      # クイックスタートの 3 本 (quick_start, quick_start_c, quick_start_f) だけ
#   生成物: examples/build/{quick_start, quick_start_c, quick_start_f, example_pcg_ks, sample_c, sample_f}、
#           ログは examples/build/<対象>.log、成功したらビルドの設定とソースの記録 examples/build/build_stamp.txt。
#           ビルドして実行し合否まで見るのは examples/run_quick_checks.sh (記録が今と同じならビルドを飛ばす)。
#   bash examples/build_examples.sh --print-stamp [quick]   ビルドせずに今の設定とソースの記録を出す
#
#   環境変数で変えられるもの (括弧内は既定値):
#     SYCL_TARGET   SYCL のターゲット (--acpp-targets=omp)。GPU なら例えば --acpp-targets=cuda:sm_70
#     CXXFLAGS      C++ の追加フラグ (-O3)。CPU では -D_BLOCKING_LOOP_DOT_=1 を足すと縮約が速い
#     LAPACK_INC    LAPACKE のインクルード (-I/usr/include/openblas)
#     LAPACK_LIB    LAPACK のリンク (-lopenblas)
#     MPICXX / MPICC / MPIFC   MPI のコンパイララッパ (mpicxx / mpicc / mpif90)
#     FC_LINK       Fortran の実行時ライブラリ (既定は "$(mpif90 --showme:link) -lgfortran"。Open MPI + gfortran 用)
#     WRAPPER_FLAGS C / Fortran ラッパの追加フラグ (-DMAX_ITERATION=10000。旧 API の反復上限)
#     PARCEL_MPI    0 にすると MPI なしのビルド (既定 1)。-DPARCEL_MPI=0 を付け、MPICXX / MPICC / MPIFC の
#                   既定を acpp / gcc / gfortran にする (MPI のコンパイララッパを使わない)。
#                   Fortran のサンプルは examples/sample_f_mpi_stub.f90 (1 プロセス用の mpi モジュール) と組む。
#                   実行は mpirun なしで ./examples/build/quick_start など。
#     BUILD_JOBS    同時に走らせるコンパイルの数 (既定は同時に走らせられる対象の数 (全部 5、quick 4) とコア数の小さい方)。
#                   1 なら順に。SYCL のコンパイル 1 本は数 GB のメモリを使うことがある
#
#   C / Fortran のサンプルは、ラッパ (capi/wrapper_parcel_sycl.cpp) を SYCL コンパイラで、
#   main を C / Fortran のコンパイラで作り、SYCL ランタイムが要るのでリンクは SYCL コンパイラで行う。
#   AdaptiveCpp (clang) は PIE でリンクするので、gcc / gfortran のオブジェクトは -fPIE で作る。
#   時間がかかるのは SYCL のコンパイル (C++ のサンプルとラッパ) で、これらを同時に走らせる (1 段目)。
#   C / Fortran の main のコンパイルも 1 段目に入れ (Fortran はモジュールの順があるので 1 本の流れ)、2 段目でリンクする。
set -u
cd "$(dirname "$0")/.." || exit 1
PRINT_STAMP=0
[ "${1:-}" = --print-stamp ] && { PRINT_STAMP=1; shift; }
TARGETS=${1:-all}
case "${TARGETS}" in all|quick) ;; *) echo "対象は all か quick: ${TARGETS}" >&2; exit 2 ;; esac
SYCL_TARGET=${SYCL_TARGET:---acpp-targets=omp}
CXXFLAGS=${CXXFLAGS:--O3}
LAPACK_INC=${LAPACK_INC:--I/usr/include/openblas}
LAPACK_LIB=${LAPACK_LIB:--lopenblas}
PARCEL_MPI=${PARCEL_MPI:-1}
if [ "${PARCEL_MPI}" = 0 ]; then
  MPICXX=${MPICXX:-acpp}
  MPICC=${MPICC:-gcc}
  MPIFC=${MPIFC:-gfortran}
  NOMPI="-DPARCEL_MPI=0"
else
  MPICXX=${MPICXX:-mpicxx}
  MPICC=${MPICC:-mpicc}
  MPIFC=${MPIFC:-mpif90}
  NOMPI=""
fi
WRAPPER_FLAGS=${WRAPPER_FLAGS:--DMAX_ITERATION=10000}
if [ -z "${FC_LINK+x}" ]; then
  if [ "${PARCEL_MPI}" = 0 ]; then
    FC_LINK="-L$(dirname "$(${MPIFC} -print-file-name=libgfortran.so)") -lgfortran"
  else
    FC_LINK="$(${MPIFC} --showme:link 2>/dev/null) -lgfortran"
  fi
fi

B=examples/build
STAMP=${B}/build_stamp.txt

# ビルドの設定とソースの記録 (run_quick_checks.sh が、前回のビルドと同じかを見る。targets の行は比べない)
stamp () {
  echo "targets=${TARGETS}"
  local v w
  for v in SYCL_TARGET CXXFLAGS LAPACK_INC LAPACK_LIB MPICXX MPICC MPIFC FC_LINK WRAPPER_FLAGS PARCEL_MPI \
           OMPI_CXX OMPI_CC OMPI_FC MPICH_CXX MPICH_CC MPICH_FC I_MPI_CXX I_MPI_CC I_MPI_F90; do
    echo "${v}=${!v-}"
  done
  for v in MPICXX MPICC MPIFC; do       # コンパイラの実体の場所 (PATH が変わったとき)
    w=${!v}; echo "path_${v}=$(command -v ${w%% *})"
  done
  # ヘッダ・C / Fortran の API・サンプルのソースとこのスクリプトの中身
  echo "sources=$(find Include capi examples/*.cpp examples/*.c examples/*.f90 examples/build_examples.sh -type f 2>/dev/null \
                  | LC_ALL=C sort | xargs cat | cksum)"
}
if [ "${PRINT_STAMP}" = 1 ]; then stamp; exit 0; fi

if [ "${TARGETS}" = quick ]; then
  EXES="quick_start quick_start_c quick_start_f"; NPAR=4
else
  EXES="quick_start quick_start_c quick_start_f example_pcg_ks sample_c sample_f"; NPAR=5
fi
NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
JOBS=${BUILD_JOBS:-$(( NPAR < NCPU ? NPAR : NCPU ))}
[ "${JOBS}" -ge 1 ] 2>/dev/null || JOBS=1

mkdir -p ${B}
rm -f ${STAMP} ${B}/*.o ${B}/*.mod ${B}/*.status
for f in ${EXES}; do rm -f ${B}/${f}; done
[ -n "${NOMPI}" ] && echo "=== MPI なしのビルド (PARCEL_MPI=0) ==="
echo "=== 対象 ${TARGETS} (${SYCL_TARGET})、同時に ${JOBS} 本 ==="

STEPS=""
step () {   # $1=名前 $2...=コマンド。結果は ${B}/<名前>.status (0 / 1)、出力は ${B}/<名前>.log
  local NAME=$1; shift
  if "$@" > ${B}/${NAME}.log 2>&1; then echo 0 > ${B}/${NAME}.status; else echo 1 > ${B}/${NAME}.status; fi
}
throttle () {   # 走っているジョブが JOBS 本より少なくなるまで待つ (bash 3.2 (macOS) でも動く形)
  while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do sleep 1; done
}
launch () {         # $1=記録する名前 (空白区切りで複数可) $2...=コマンド (関数でもよい)。JOBS 本までは同時に走らせる
  STEPS="${STEPS} $1"; shift
  if [ "${JOBS}" = 1 ]; then "$@"; else throttle; "$@" & fi
}
report () {     # 終わった段の結果を、名前の順に表示する
  local NAME
  for NAME in "$@"; do
    if [ "$(cat ${B}/${NAME}.status 2>/dev/null)" = 0 ]; then echo "  [OK]   ${NAME}"
    else
      echo "  [FAIL] ${NAME} (${B}/${NAME}.log)"
      grep -E "error" ${B}/${NAME}.log 2>/dev/null | head -5 | sed 's/^/         /'
    fi
  done
}

FSTUB=""
[ -n "${NOMPI}" ] && FSTUB=${B}/sample_f_mpi_stub.o
fortran_objs () {   # モジュールの順があるので 1 本の流れで (代用 mpi → parcel_sycl_mod → main)
  if [ -n "${NOMPI}" ]; then
    step sample_f_mpi_stub.o ${MPIFC} -O2 -fPIE -J${B} -c examples/sample_f_mpi_stub.f90 -o ${B}/sample_f_mpi_stub.o
  fi
  step parcel_sycl_mod.o ${MPIFC} -O2 -fPIE -J${B} -c capi/parcel_sycl_mod.f90 -o ${B}/parcel_sycl_mod.o
  step quick_start_f.o ${MPIFC} -O2 -fPIE -I${B} -c examples/quick_start_f.f90 -o ${B}/quick_start_f.o
  if [ "${TARGETS}" = all ]; then
    step sample_f.o  ${MPIFC} -O2 -fPIE -I${B} -c examples/sample_f.f90 -o ${B}/sample_f.o
    step problem.o   ${MPIFC} -O2 -fPIE -c examples/problem.f90 -o ${B}/problem.o
  fi
}
c_objs () {
  step quick_start_c.o ${MPICC} -O2 -fPIE ${NOMPI} -Icapi -c examples/quick_start_c.c -o ${B}/quick_start_c.o
  [ "${TARGETS}" = all ] && step sample_c.o ${MPICC} -O2 -fPIE ${NOMPI} -Icapi -c examples/sample_c.c -o ${B}/sample_c.o
  return 0
}

# 1 段目: SYCL のコンパイル (C++ のサンプル、ラッパ) と C / Fortran の main
FNAMES="parcel_sycl_mod.o quick_start_f.o"
[ -n "${NOMPI}" ] && FNAMES="sample_f_mpi_stub.o ${FNAMES}"
CNAMES="quick_start_c.o"
[ "${TARGETS}" = all ] && { FNAMES="${FNAMES} sample_f.o problem.o"; CNAMES="${CNAMES} sample_c.o"; }
echo "--- コンパイル"
launch wrapper step wrapper ${MPICXX} ${CXXFLAGS} ${NOMPI} ${SYCL_TARGET} ${WRAPPER_FLAGS} ${LAPACK_INC} \
                 -c capi/wrapper_parcel_sycl.cpp -o ${B}/wrapper_parcel_sycl.o
launch quick_start step quick_start ${MPICXX} ${CXXFLAGS} ${NOMPI} ${SYCL_TARGET} ${LAPACK_INC} -o ${B}/quick_start examples/quick_start.cpp ${LAPACK_LIB}
if [ "${TARGETS}" = all ]; then
  launch example_pcg_ks step example_pcg_ks ${MPICXX} ${CXXFLAGS} ${NOMPI} ${SYCL_TARGET} ${LAPACK_INC} \
                 -o ${B}/example_pcg_ks examples/example_pcg_ks.cpp ${LAPACK_LIB}
fi
launch "${CNAMES}" c_objs
launch "${FNAMES}" fortran_objs
wait
report ${STEPS}

# 2 段目: C / Fortran のリンク
STEPS=""
echo "--- リンク"
launch quick_start_c step quick_start_c ${MPICXX} ${SYCL_TARGET} -o ${B}/quick_start_c ${B}/quick_start_c.o ${B}/wrapper_parcel_sycl.o ${LAPACK_LIB} -lm
launch quick_start_f step quick_start_f ${MPICXX} ${SYCL_TARGET} -o ${B}/quick_start_f ${B}/quick_start_f.o ${FSTUB} ${B}/parcel_sycl_mod.o \
                 ${B}/wrapper_parcel_sycl.o ${FC_LINK} ${LAPACK_LIB}
if [ "${TARGETS}" = all ]; then
  launch sample_c step sample_c ${MPICXX} ${SYCL_TARGET} -o ${B}/sample_c ${B}/sample_c.o ${B}/wrapper_parcel_sycl.o ${LAPACK_LIB} -lm
  launch sample_f step sample_f ${MPICXX} ${SYCL_TARGET} -o ${B}/sample_f ${B}/sample_f.o ${FSTUB} ${B}/parcel_sycl_mod.o \
                 ${B}/wrapper_parcel_sycl.o ${FC_LINK} ${LAPACK_LIB}
fi
wait
report ${STEPS}

# 成果物で確かめる (コマンドの終了コードだけで判断しない)
NG=$(cat ${B}/*.status 2>/dev/null | grep -c 1)
for f in ${EXES}; do
  [ -x ${B}/${f} ] || { echo "  MISSING ${B}/${f}"; NG=$((NG+1)); }
done
[ "${NG}" = 0 ] && stamp > ${STAMP}
echo "=== 失敗 ${NG} 件 ==="
exit $(( NG > 0 ))
