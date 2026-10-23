# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""sample_py.py : Python パッケージ parcel の検証 (C の examples/sample_c.c / Fortran の sample_f.f90 に準じる)

回帰試験の lang_py が使う。
問題・項目・判定基準・設定は sample_c.c と同じ:

  3 次元 7 点ポアソン行列 (中心 -6, 隣接 1, ディリクレ境界, NX^3 行。NX 既定 16)
  行はランク r に重み (r+1) で不均等に配る (np=3 なら 1:2:3、z 面の途中で切れる)
  右辺 b_g = 1 + 0.1 sin(0.37 g)  (g は大域の行番号)
  線形 6 項目 (pcg / pbicgstab / pgmres / pcagmres / pcagmres(implicit) / pcagmres(impl,sym)):
      tol 1e-9, maxiter 10000, ブロックヤコビ, ||b - Ax|| / ||b|| <= 1e-7
  固有値 4 項目 (power / inverse_power / ks_arnoldi / ks_arnoldi(shift-inv)): 解析固有値
      λ(i,j,k) = -2 Σ_d (1 - cos(k_d π / (NX+1)))  との相対誤差 <= 1e-6 (ks は |Im|, 残差も見る)
  各ソルバの PASS には、ライブラリが返す収束状態が収束であることも含める。
  最後の項目は sample_c の旧 API (run_parcel_sycl_c) の代わりに、設定をすべて省いた
  parcel.solve(A, b) (method の既定 pcg、tol などはライブラリの既定) を同じ残差の基準で見る
  (Python には旧 API が無い。「最小限の呼び出しで解ける」ことを確かめる役目は同じ)。

環境変数で与えたときだけ先頭に足す項目 (回帰試験の生成器が与える):
  PARCEL_EXPECT_MPI=0|1   -> has_mpi    : ネイティブ層の MPI の有無が期待どおりか
  PARCEL_EXPECT_DEVICE=.. -> device     : デバイス名が候補 ("|" 区切り) のどれかを含むか (全ランク)
その次に必ず出す項目:
  version : ネイティブ層の版 parcel.library_version() (Include/parcel_version.hpp) がパッケージの版
            parcel.__version__ と同じか。PARCEL_EXPECT_VERSION を与えればその値とも比べる
            (回帰試験の生成器が Include/parcel_version.hpp から与える。C / Fortran の version と同じ)

出力は rank 0 だけ: "[sample_py] <名前> PASS|FAIL  rc=<n>  <詳細>" (ソルバの項目は次の行に収束状態)、
最後に "[sample_py] ALL PASS (0 failed)" または "[sample_py] FAILED (<n> failed)"。失敗があれば終了コード 1。
判定は全ランクで揃える (comm.allreduce の min)。検証の演算はライブラリとは独立 (numpy / scipy)。

    python tests/regression/sample_py.py [NX]                 (MPI なし、または MPI ありの 1 プロセス)
    mpirun -n 4 python tests/regression/sample_py.py [NX]

PARCEL_TEST_FROM_SOURCE=1 ならパッケージの python/ を読む (PARCEL_PY_LIB でネイティブ層を指定)。
未設定なら、インストールされた parcel (pip や spack の py-parcel-sycl) を読む。
"""
import ctypes
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

if os.environ.get("PARCEL_TEST_FROM_SOURCE") == "1":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import parcel  # noqa: E402

TAG = "[sample_py]"
comm = parcel.Comm()
RANK, SIZE = comm.rank, comm.size

# ネイティブ層 (C の stdio) が verbose で書いた分を、こちらの行より先に出す
#   (ファイルへのリダイレクトでは C の stdout は満杯になるまで溜まり、行の途中で混ざることがある)
try:
    _libc = ctypes.CDLL(None)
except OSError:  # pragma: no cover
    _libc = None


def flush_all():
    sys.stdout.flush()
    if _libc is not None:
        _libc.fflush(None)


n_fail = 0


def report(name, rc, ok, detail, info=None):
    """1 項目の判定と表示。info (SolveInfo / EigInfo) を与えれば収束も判定に含める。"""
    global n_fail
    if rc != 0:
        ok = False
    if info is not None and not (info.reason_code > 0):
        ok = False
    ok = comm.allreduce(1.0 if ok else 0.0, "min") > 0.5    # 全ランクで揃える
    if not ok:
        n_fail += 1
    flush_all()
    if RANK == 0:
        print("%s %-22s %s  rc=%d  %s" % (TAG, name, "PASS" if ok else "FAIL", rc, detail))
        if info is not None:
            print("             converged_reason = %d (> 0 で収束)  its = %d  residual = %.3e  true_residual = %.3e"
                  "  ks_nconv = %d  linsolve_failures = %d"
                  % (info.reason_code, info.iterations, info.residual, info.true_residual,
                     getattr(info, "n_converged", 0), getattr(info, "inner_failures", 0)))
        sys.stdout.flush()


def error_rc(e):
    """例外 -> 表示用の rc (ライブラリの誤りの番号。分からなければ -1)"""
    code = getattr(e, "code", None)
    return int(code) if isinstance(code, int) and code != 0 else -1


# ---- 行列: sample_c の poisson_rows と同じ (g = ix + nx*iy + nx^2*iz、中心 -6、隣接 1) ----
def poisson3d(nx):
    I = sp.identity(nx, format="csr")
    T = sp.diags([np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1], format="csr")
    return (sp.kron(sp.kron(I, I), T) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(T, I), I)).tocsr()


# ---- 検証用の分散ベクトル演算 (ライブラリとは独立) ----
def gather(x_local):
    """分散ベクトルの全体 (全ランクが持つ)。0 を足すだけなので値は変わらない。"""
    full = np.zeros(N_GLOBAL)
    full[SL] = x_local
    if SIZE > 1:
        for i in range(N_GLOBAL):
            full[i] = comm.allreduce(full[i])
    return full


def spmv(x_local):
    return G_LOCAL @ gather(x_local)


def dot(a, b):
    return comm.allreduce(float(np.dot(a, b)))


# ---- 解析固有値: 全固有値を |λ-σ| で並べる (desc=True で降順) ----
def analytic_eigs(nx, sigma, desc):
    c = 2.0 * (1.0 - np.cos(np.arange(1, nx + 1) * math.pi / (nx + 1)))
    lam = -(c[:, None, None] + c[None, :, None] + c[None, None, :]).ravel()
    key = np.abs(lam - sigma)
    order = np.argsort(-key if desc else key, kind="stable")
    return lam[order]


def sort_by_key(vals, sigma, desc):
    key = np.abs(vals - sigma)
    return vals[np.argsort(-key if desc else key, kind="stable")]


# ================================================================ main
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_GLOBAL = NX ** 3

# 不均等分割: ランク r の重み r+1 (sample_c と同じ整数の計算)
W = SIZE * (SIZE + 1) // 2
counts, displs, acc = [], [], 0
for r in range(SIZE):
    s0 = N_GLOBAL * acc // W
    acc += r + 1
    e0 = N_GLOBAL * acc // W
    displs.append(s0)
    counts.append(e0 - s0)

G = poisson3d(NX)
A = parcel.DistributedCSR.from_global(G, comm, row_start=displs[RANK], n_local=counts[RANK])
SL = A.local_slice()
G_LOCAL = G[SL, :]

if RANK == 0:
    print("%s parcel %s from %s" % (TAG, parcel.__version__, os.path.dirname(parcel.__file__)))
    print("%s %s, OMP_NUM_THREADS=%s" % (TAG, parcel.build_info(), os.environ.get("OMP_NUM_THREADS")))
    print("%s NX=%d n_global=%d nprocs=%d rows/rank: %s"
          % (TAG, NX, N_GLOBAL, SIZE, " ".join(str(c) for c in counts)))
    sys.stdout.flush()

# ---- 環境の確認 (環境変数で与えたときだけ) ----
expect_mpi = os.environ.get("PARCEL_EXPECT_MPI")
if expect_mpi is not None:
    want = expect_mpi == "1"
    ok = parcel.has_mpi() == want and (want or (RANK, SIZE) == (0, 1))
    report("has_mpi", 0, ok, "has_mpi() = %s (expect %s)  ranks = %d" % (parcel.has_mpi(), want, SIZE))
dev = parcel.device_name()
expect_dev = os.environ.get("PARCEL_EXPECT_DEVICE")
if expect_dev:
    ok = any(k and k in dev for k in expect_dev.split("|"))
    report("device", 0, ok, "rank 0: %s (expect %r)" % (dev, expect_dev))
elif RANK == 0:
    print("%s device: %s" % (TAG, dev))
    sys.stdout.flush()

# ---- 版 ----
lib_ver = parcel.library_version()
expect_ver = os.environ.get("PARCEL_EXPECT_VERSION")
ok = lib_ver == parcel.__version__ and (not expect_ver or lib_ver == expect_ver)
report("version", 0, ok, "library_version() = %s  __version__ = %s  (expect %s)"
       % (lib_ver, parcel.__version__, expect_ver or "-"))

# ================= 線形ソルバ =================
LIN_TOL, LIN_CHECK = 1.0e-9, 1.0e-7
b_full = 1.0 + 0.1 * np.sin(0.37 * np.arange(N_GLOBAL))
b = b_full[SL].copy()
bnorm = math.sqrt(dot(b, b))


def lin_residual(x):
    r = b - spmv(x)
    return math.sqrt(dot(r, r)) / bnorm


# sample_c と同じ設定 (pgmres: restart 8・MGS、pcagmres: s 8・t 1・MGS。最後の 2 つは Q を陽に作らない版)
LINEAR = [
    ("pcg", dict(method="pcg")),
    ("pbicgstab", dict(method="pbicgstab")),
    ("pgmres", dict(method="pgmres", restart=8, orthogonalization="mgs")),
    ("pcagmres", dict(method="pcagmres", s=8, t=1, qr="mgs")),
    # CholeskyQR。t = 2 でブロック (k > 0) の直交化も通す
    ("pcagmres(implicit)", dict(method="pcagmres", s=8, t=2, q_mode="implicit", qr="cholesky_qr")),
    # モーメント法 (implicit 専用): A が対称 (ここでは成り立つ)、前処理なし、s が偶数
    ("pcagmres(impl,sym)", dict(method="pcagmres", s=8, t=2, q_mode="implicit", qr="cholesky_qr_moment",
                                precond=None)),
]
for i, (name, kw) in enumerate(LINEAR):
    kw = dict(kw)
    kw.setdefault("precond", "block_jacobi")
    t0 = time.perf_counter()
    try:
        x, info = parcel.solve(A, b, tol=LIN_TOL, maxiter=10000, verbose=(i == 0), **kw)
        rc = 0
    except Exception as e:  # noqa: BLE001  (ライブラリの誤りは全ランクで同じ例外)
        flush_all()
        report(name, error_rc(e), False, "%s: %s" % (type(e).__name__, e))
        continue
    t1 = time.perf_counter()
    rel = lin_residual(x)
    report(name, rc, rel <= LIN_CHECK, "||b-Ax||/||b|| = %.3e (<= %.0e)  %.2f s" % (rel, LIN_CHECK, t1 - t0), info)

# ================= 固有値ソルバ =================
# ---- power: 絶対値最大 ----
try:
    t0 = time.perf_counter()
    vals, vecs, info = parcel.eigs(A, method="power", tol=1.0e-10, maxiter=20000)
    t1 = time.perf_counter()
    lam = vals[0].real
    ref = analytic_eigs(NX, 0.0, True)[0]
    err = abs(lam - ref) / abs(ref)
    report("power", 0, err <= 1.0e-6, "lambda = %.12f  exact = %.12f  rel.err = %.2e (<= 1e-6)  %.2f s"
           % (lam, ref, err, t1 - t0), info)
except Exception as e:  # noqa: BLE001
    flush_all()
    report("power", error_rc(e), False, "%s: %s" % (type(e).__name__, e))

# ---- inverse_power: 絶対値最小 (内部 pgmres + ブロックヤコビ) ----
try:
    t0 = time.perf_counter()
    vals, vecs, info = parcel.eigs(A, method="inverse_power", tol=1.0e-10, maxiter=200,
                                   inner_method="pgmres", inner_tol=1.0e-11, inner_maxiter=5000,
                                   inner_restart=30, inner_precond="block_jacobi")
    t1 = time.perf_counter()
    lam = vals[0].real
    ref = analytic_eigs(NX, 0.0, False)[0]
    err = abs(lam - ref) / abs(ref)
    v = vecs[:, 0]
    r = spmv(v) - lam * v
    res = math.sqrt(dot(r, r)) / (abs(lam) * math.sqrt(dot(v, v)))     # 固有対の残差 (参考)
    report("inverse_power", 0, err <= 1.0e-6,
           "lambda = %.12f  exact = %.12f  rel.err = %.2e (<= 1e-6)  resid = %.1e  %.2f s"
           % (lam, ref, err, res, t1 - t0), info)
except Exception as e:  # noqa: BLE001
    flush_all()
    report("inverse_power", error_rc(e), False, "%s: %s" % (type(e).__name__, e))

# ---- ks_arnoldi: (a) 絶対値最大側 nev 個, (b) shift-invert σ=-0.15 近傍 nev 個 ----
#   固有値は縮重しているので「各 Ritz 値が最も近い解析固有値と一致するか」、最良の 1 個が解析値の端と
#   一致するか、|Im| と固有対の残差が小さいかで判定する (sample_c と同じ)。多重度の再現は参考表示。
NEV, SIGMA = 4, -0.15
for mode in (0, 1):
    name = "ks_arnoldi" if mode == 0 else "ks_arnoldi(shift-inv)"
    kw = dict(method="ks_arnoldi", tol=1.0e-8, m=30, k_keep=10, nev=NEV, max_restart=100)
    if mode == 1:
        # precon_spectrum=False: ラッパが (A - σI) を組む。内部 pgmres の上限 5000 を明示 (sample_c と同じ)
        kw.update(sigma=SIGMA, inner_tol=1.0e-11, inner_maxiter=5000, inner_restart=30,
                  inner_precond="block_jacobi")
    try:
        t0 = time.perf_counter()
        vals, vecs, info = parcel.eigs(A, **kw)
        t1 = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        flush_all()
        report(name, error_rc(e), False, "%s: %s" % (type(e).__name__, e))
        continue
    key = 0.0 if mode == 0 else SIGMA
    desc = (mode == 0)
    ref = analytic_eigs(NX, key, desc)
    got = vals[:NEV].real.copy()
    maxim = float(np.max(np.abs(vals[:NEV].imag)))
    maxres = maxerr = 0.0
    for j in range(NEV):
        v = vecs[:, j]
        r = spmv(v) - got[j] * v
        rj = math.sqrt(dot(r, r)) / (abs(got[j]) * math.sqrt(dot(v, v)))
        if not (rj <= maxres):
            maxres = rj
        e = np.min(np.abs(got[j] - ref)) / abs(got[j])
        if not (e <= maxerr):
            maxerr = e                                   # NaN も拾う
    got_s = sort_by_key(got, key, desc)
    err_top = abs(got_s[0] - ref[0]) / abs(ref[0])
    n_multi = int(np.sum(np.abs(got_s - ref[:NEV]) <= 1.0e-6 * np.abs(ref[:NEV])))
    ok = (maxerr <= 1.0e-6) and (err_top <= 1.0e-6) and (maxim <= 1.0e-8) and (maxres <= 1.0e-6)
    report(name, 0, ok,
           "%d eigs: nearest rel.err = %.2e, extreme rel.err = %.2e (<= 1e-6)  max|Im| = %.1e  resid = %.1e (<= 1e-6)  %.2f s"
           % (NEV, maxerr, err_top, maxim, maxres, t1 - t0), info)
    if RANK == 0:
        for j in range(NEV):
            print("             %2d  %.12f  exact(sorted) %.12f" % (j, got_s[j], ref[j]))
        print("             multiplicity reproduced: %d / %d (reference only)" % (n_multi, NEV))
        sys.stdout.flush()

# ================= 既定値だけの呼び出し (sample_c の旧 API の項目の代わり) =================
try:
    t0 = time.perf_counter()
    x, info = parcel.solve(A, b)
    t1 = time.perf_counter()
    rel = lin_residual(x)
    report("pcg(defaults)", 0, rel <= LIN_CHECK,
           "||b-Ax||/||b|| = %.3e (<= %.0e)  %.2f s" % (rel, LIN_CHECK, t1 - t0), info)
except Exception as e:  # noqa: BLE001
    flush_all()
    report("pcg(defaults)", error_rc(e), False, "%s: %s" % (type(e).__name__, e))

flush_all()
if RANK == 0:
    print("%s %s (%d failed)" % (TAG, "ALL PASS" if n_fail == 0 else "FAILED", n_fail))
    sys.stdout.flush()
sys.exit(0 if n_fail == 0 else 1)
