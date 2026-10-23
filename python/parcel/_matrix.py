# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""各ランクが担当行を持つ分散 CSR 行列。"""
import numpy as np

from ._comm import as_comm


class DistributedCSR:
    """MPI で行ブロックに分割した疎行列のうち, このランクの担当行。

    indptr, indices, data は担当行の CSR (scipy.sparse.csr_matrix と同じ並び)。
    indptr は 0 始まりの局所行ポインタ, indices は大域の 0 始まり列番号。
    担当行は大域の [row_start, row_start + n_local) で, 全ランクの行ブロックが
    [0, n_global) を隙間・重複なく覆う必要がある (ランクの並び順は問わない)。
    配列は内部で int64 / float64 の連続配列に複製するので, 呼び出し後に変更してよい。
    """

    def __init__(self, indptr, indices, data, row_start, n_global, comm=None):
        self.comm = as_comm(comm)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64).copy()
        self.indices = np.ascontiguousarray(indices, dtype=np.int64).copy()
        self.data = np.ascontiguousarray(data, dtype=np.float64).copy()
        self.row_start = int(row_start)
        self.n_global = int(n_global)
        if self.indptr.ndim != 1 or self.indptr.size < 2:
            raise ValueError("indptr must be a 1-D array of length n_local + 1 (n_local >= 1)")
        if self.indices.shape != self.data.shape or self.indices.ndim != 1:
            raise ValueError("indices and data must be 1-D arrays of the same length")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.indices.size:
            raise ValueError("indptr[0] must be 0 and indptr[-1] must equal len(indices)")

    @property
    def n_local(self):
        return self.indptr.size - 1

    @property
    def nnz(self):
        return self.indices.size

    @property
    def shape(self):
        return (self.n_global, self.n_global)

    @classmethod
    def from_scipy(cls, local_rows, row_start, n_global=None, comm=None):
        """scipy.sparse の行列 (担当行だけ, 列は大域) から作る。n_global を省くと列数を使う。"""
        m = local_rows.tocsr()
        if n_global is None:
            n_global = m.shape[1]
        return cls(m.indptr, m.indices, m.data, row_start, n_global, comm)

    @classmethod
    def from_global(cls, global_matrix, comm=None, row_start=None, n_local=None):
        """全体の行列 (全ランクが同じものを持つ) から, このランクの担当行を切り出す。

        小さな問題や試験向け。row_start, n_local を省くと partition_rows() の割り当てを使う。
        """
        from ._comm import partition_rows
        c = as_comm(comm)
        g = global_matrix.tocsr()
        if g.shape[0] != g.shape[1]:
            raise ValueError("the matrix must be square")
        if row_start is None or n_local is None:
            row_start, n_local = partition_rows(g.shape[0], c)
        return cls.from_scipy(g[row_start:row_start + n_local, :], row_start, g.shape[0], c)

    def local_slice(self):
        """全体のベクトルからこのランクの部分を取り出すスライス。"""
        return slice(self.row_start, self.row_start + self.n_local)

    def __repr__(self):
        return ("parcel.DistributedCSR(n_global=%d, rows %d..%d on rank %d, nnz_local=%d)"
                % (self.n_global, self.row_start, self.row_start + self.n_local - 1,
                   self.comm.rank, self.nnz))
