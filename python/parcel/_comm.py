# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""MPI コミュニケータの受け渡し。"""
import ctypes

from . import _native


class Comm:
    """PARCEL に渡す MPI コミュニケータ。

    Comm() は MPI_COMM_WORLD を表す (mpi4py が無くても使える)。
    Comm(c) には mpi4py のコミュニケータを渡せる。
    MPI なしでビルドしたネイティブ層では Comm() だけを使い, rank は 0, size は 1 になる。
    """

    def __init__(self, mpi4py_comm=None):
        _native.lib()
        self._mpi4py = mpi4py_comm
        self._handle = None
        if mpi4py_comm is not None:
            if not _native.has_mpi():
                raise TypeError("the native library was built without MPI (PARCEL_MPI=0); use parcel.Comm()")
            try:
                from mpi4py import MPI
            except ImportError as e:  # pragma: no cover
                raise TypeError("an mpi4py communicator was given but mpi4py is not importable") from e
            if not isinstance(mpi4py_comm, MPI.Comm):
                raise TypeError("comm must be an mpi4py communicator or None")
            # MPI_Comm を保持している場所のアドレス (C 側は MPI_Comm * として読む)
            self._handle = ctypes.c_void_p(MPI._addressof(mpi4py_comm))

    @property
    def handle(self):
        return self._handle

    @property
    def rank(self):
        return _native.lib().parcel_py_comm_rank(self._handle)

    @property
    def size(self):
        return _native.lib().parcel_py_comm_size(self._handle)

    def allreduce(self, value, op="sum"):
        """全ランクの和 ("sum"), 最大 ("max"), 最小 ("min")。集団操作 (全ランクで呼ぶ)。"""
        ops = {"sum": 0, "max": 1, "min": 2}
        if op not in ops:
            raise ValueError("op must be 'sum', 'max' or 'min'")
        return _native.lib().parcel_py_allreduce(self._handle, float(value), ops[op])

    def __repr__(self):
        if not _native.has_mpi():
            name = "no MPI"
        else:
            name = "MPI_COMM_WORLD" if self._mpi4py is None else repr(self._mpi4py)
        return "parcel.Comm(%s, rank=%d, size=%d)" % (name, self.rank, self.size)


def as_comm(comm):
    if comm is None:
        return Comm()
    if isinstance(comm, Comm):
        return comm
    return Comm(comm)


def partition_rows(n_global, comm=None):
    """n_global 行を各ランクに連続した行ブロックとして均等に割り当てる。

    返り値は (row_start, n_local)。端数は先頭のランクから 1 行ずつ割り当てる。
    """
    c = as_comm(comm)
    p, r = c.size, c.rank
    n_global = int(n_global)
    if n_global < p:
        raise ValueError("n_global (%d) must be at least the number of ranks (%d)" % (n_global, p))
    base, extra = divmod(n_global, p)
    n_local = base + (1 if r < extra else 0)
    row_start = r * base + min(r, extra)
    return row_start, n_local
