# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# Wisteria-Aquarius: spack で作った GCC 12 を PATH に置く (利用者の C / Fortran のプログラムを同じ gfortran でコンパイルする。
#   Fortran のモジュール parcel_sycl_mod.mod は gfortran の版が違うと読めない)
spack load gcc@12 > /dev/null 2>&1
