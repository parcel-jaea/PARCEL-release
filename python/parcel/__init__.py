# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""PARCEL (SYCL 版) の Python インタフェース。

    import parcel
    A = parcel.DistributedCSR.from_global(A_global)      # このランクの担当行を切り出す
    x, info = parcel.solve(A, b[A.local_slice()], method="pcg", precond="jacobi", tol=1e-10)
    vals, vecs, info = parcel.eigs(A, method="ks_arnoldi")

MPI 並列では mpirun -n 4 python script.py のように起動し, 全ランクが同じ呼び出しを行う。
MPI なしでビルドしたネイティブ層 (PARCEL_MPI=0) では python script.py で 1 プロセスとして動く。
C / Fortran の API とは独立したネイティブ層 (libparcel_py.so) を使う。精度は倍精度だけ。
"""
from ._comm import Comm, partition_rows
from ._matrix import DistributedCSR
from ._native import build_info, device_name, has_mpi, library_version
from ._solvers import EigInfo, SolveInfo, eigs, solve
from .errors import NotBuiltError, ParcelError

# パッケージの版。本体 (Include/parcel_version.hpp) と同じにする (tests/check_version.py で照合。
#   ネイティブ層の版は library_version())
__version__ = "1.2.0"

__all__ = ["Comm", "partition_rows", "DistributedCSR", "solve", "eigs",
           "SolveInfo", "EigInfo", "ParcelError", "NotBuiltError", "build_info", "device_name", "has_mpi",
           "library_version"]
