# quick_start.py : Python のパッケージ parcel で 1 次元 Laplace 作用素の三重対角行列の連立一次方程式を pcg で解く最小の例
#   MPI なしの版: python quick_start.py, MPI ありの版: mpirun -n 4 python quick_start.py (同じスクリプト)
# ネイティブ層を pip でなく直接ビルドした場合: PYTHONPATH=python PARCEL_PY_LIB=<libparcel_py.so> python examples/quick_start.py
#
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
import numpy as np
import scipy.sparse as sp
import parcel

n_global = 1000
G = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n_global, n_global), format="csr")
A = parcel.DistributedCSR.from_global(G)          # このランクの担当行を切り出す (MPI なしでは全行)
b = np.ones(A.n_local)
x, info = parcel.solve(A, b, method="pcg", tol=1e-9, maxiter=10000)   # 前処理なし, 倍精度

# 厳密解との比較. 行列は境界の外の値を 0 とした (Dirichlet 境界) 差分 -x[i-1] + 2 x[i] - x[i+1] = 1 なので,
#   厳密解は x[i] = (i + 1) (n_global - i) / 2 (i は大域 0 始まり). 全成分の相対誤差の最大を各ランクの担当行で求め,
#   ランクの最大を集める. しきい値 1e-6 の根拠は quick_start.cpp を参照 (条件数 約 4.1e5, tolerance 1e-9)
err_limit = 1.0e-6
i = np.arange(A.row_start, A.row_start + A.n_local, dtype=float)
exact = 0.5 * (i + 1.0) * (n_global - i)
rel = np.abs(x - exact) / exact
rel[~np.isfinite(rel)] = np.inf                   # NaN は無限大として扱う (不合格にする)
err_max = A.comm.allreduce(float(rel.max()), op="max")
ok = info.converged and err_max <= err_limit
if A.comm.rank == 0:
    print(f"reason = {info.reason}, iterations = {info.iterations}, "
          f"residual = {info.residual:.3e}, true residual = {info.true_residual:.3e}")
    print(f"x[0] = {x[0]:.6f} (exact {0.5 * n_global:.6f})")
    print(f"check: max relative error = {err_max:.3e} (limit {err_limit:.1e}) {'PASS' if ok else 'FAIL'}")
if not ok:
    raise SystemExit(1)
