# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""連立一次方程式と固有値問題のソルバ。"""
import ctypes
import math
from dataclasses import dataclass

import numpy as np

from . import _native
from ._matrix import DistributedCSR
from .errors import NotBuiltError, ParcelError

_PRECOND = {None: 0, "none": 0, "point_jacobi": 1, "block_jacobi": 2, "jacobi": 4}
_ORTH = {"cgs": 0, "mgs": 1}
# ks_arnoldi の直交化 (どちらも 2 回適用する。"cgs2" / "mgs2" も受け付ける)
_KS_ORTH = {"cgs": 0, "mgs": 1, "cgs2": 0, "mgs2": 1}
_QR = {"cgs": 0, "mgs": 1, "cholesky_qr": 2, "cholesky_qr_moment": 202}
_QMODE = {"auto": 0, "explicit": 1, "implicit": 2}
_INNER = {"pgmres": 0, "pcg": 1}
_REASONS = {1: "converged", 0: "iterating", -1: "max_iterations",
            -2: "nan_or_inf", -3: "breakdown"}


@dataclass(frozen=True)
class SolveInfo:
    """収束状態。全ランクで同じ値になる。

    reason は "converged", "max_iterations", "nan_or_inf", "breakdown" のいずれか。
    residual はソルバが収束判定に用いた相対残差, true_residual は終了時の ||b - Ax|| / ||b||
    (固有値ソルバでは NaN)。iterations の数え方はソルバによる (API リファレンスの get_iterations())。
    """
    converged: bool
    reason: str
    iterations: int
    residual: float
    true_residual: float
    reason_code: int


@dataclass(frozen=True)
class EigInfo(SolveInfo):
    """固有値ソルバの収束状態。n_converged は ks_arnoldi で判定値を満たした固有対の数。"""
    n_converged: int = 0
    inner_failures: int = 0


def _choice(name, value, table):
    if value is None:
        return -1
    key = value.lower() if isinstance(value, str) else value
    if key not in table:
        raise ValueError("%s must be one of %s (got %r)" % (name, sorted(k for k in table if k), value))
    return table[key]


def _pos_int(name, value):
    if value is None:
        return -1
    v = int(value)
    if v < 1:
        raise ValueError("%s must be >= 1 (got %r)" % (name, value))
    return v


def _pos_float(name, value):
    if value is None:
        return float("nan")
    v = float(value)
    if not (v > 0.0) or math.isinf(v):
        raise ValueError("%s must be a finite positive number (got %r)" % (name, value))
    return v


def _tristate(name, value):
    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)) or value in (0, 1):
        return int(bool(value))
    raise ValueError("%s must be True, False or None (got %r)" % (name, value))


def _apply_precond(o, precond, precond_options, inner=False):
    po = dict(precond_options or {})
    unknown = set(po) - {"iterations", "blocks", "scale_matrix", "scale_vector"}
    if unknown:
        raise ValueError("unknown precond_options: %s" % sorted(unknown))
    if inner:
        if "scale_matrix" in po or "scale_vector" in po:
            raise ValueError("scaling of the inner solver's preconditioner is chosen automatically")
        o.inner_precond = _choice("precond", precond, _PRECOND)
        o.inner_jacobi_iter = _pos_int("iterations", po.get("iterations"))
        o.inner_blocks = _pos_int("blocks", po.get("blocks"))
    else:
        o.precond = _choice("precond", precond, _PRECOND)
        o.jacobi_iter = _pos_int("iterations", po.get("iterations"))
        o.blocks = _pos_int("blocks", po.get("blocks"))
        o.scale_matrix = _tristate("scale_matrix", po.get("scale_matrix"))
        o.scale_vector = _tristate("scale_vector", po.get("scale_vector"))


def _raise(rc, err):
    msg = err.value.decode(errors="replace") or "error code %d" % rc
    if rc == _native.ERR_NOT_BUILT:
        raise NotBuiltError(msg, rc)
    if rc == _native.ERR_ARG:
        raise ValueError(msg)
    if rc == _native.ERR_OVERFLOW:
        raise OverflowError(msg)
    raise ParcelError(msg, rc)


def _matrix_args(A):
    if not isinstance(A, DistributedCSR):
        raise TypeError("A must be a parcel.DistributedCSR")
    return (A.comm.handle, A.n_local, A.row_start, A.n_global, A.nnz,
            A.indptr.ctypes.data, A.indices.ctypes.data, A.data.ctypes.data)


def solve(A, b, *, method="pcg", x0=None, tol=None, maxiter=None,
          precond=None, precond_options=None,
          restart=None, orthogonalization=None,
          s=None, t=None, qr=None, q_mode=None,
          verbose=False):
    """連立一次方程式 Ax = b を解き, (x, SolveInfo) を返す。

    A は DistributedCSR, b と x0 はこのランクの担当行の部分 (長さ A.n_local)。
    全ランクが同じ引数で呼ぶ (集団操作)。返す x も担当行の部分である。
    未指定 (None) の設定はライブラリの既定値になる (API リファレンスを参照)。

    method: "pcg", "pbicgstab", "pgmres", "pcagmres"
    precond: None (前処理なし), "point_jacobi", "block_jacobi", "jacobi"
    precond_options: {"iterations": Jacobi の反復回数, "blocks": ブロック Jacobi のブロック数,
                      "scale_matrix": bool, "scale_vector": bool}
    restart, orthogonalization ("mgs" / "cgs"): pgmres のみ
    s, t, qr ("mgs", "cgs", "cholesky_qr", "cholesky_qr_moment"), q_mode ("auto", "explicit", "implicit"):
        pcagmres のみ
    収束しなくても例外にはならない。info.converged を確かめる。
    """
    method = str(method).lower()
    if method not in _native.LINEAR:
        raise ValueError("method must be one of %s (got %r)" % (list(_native.LINEAR), method))
    if method != "pgmres" and (restart is not None or orthogonalization is not None):
        raise ValueError("restart / orthogonalization apply only to method='pgmres'")
    if method != "pcagmres" and any(v is not None for v in (s, t, qr, q_mode)):
        raise ValueError("s / t / qr / q_mode apply only to method='pcagmres'")
    if (s is None) != (t is None):
        raise ValueError("s and t must be given together")

    lib = _native.lib()
    o = _native.Options.unset()
    o.solver = _native.SOLVERS[method]
    o.verbose = 1 if verbose else 0
    o.tol = _pos_float("tol", tol)
    o.maxiter = _pos_int("maxiter", maxiter)
    _apply_precond(o, precond, precond_options)
    o.restart = _pos_int("restart", restart)
    o.orthogonalization = _choice("orthogonalization", orthogonalization, _ORTH)
    o.s, o.t = _pos_int("s", s), _pos_int("t", t)
    o.qr = _choice("qr", qr, _QR)
    o.q_mode = _choice("q_mode", q_mode, _QMODE)

    n = A.n_local
    bb = np.ascontiguousarray(b, dtype=np.float64)
    if bb.shape != (n,):
        raise ValueError("b must have shape (A.n_local,) = (%d,)" % n)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    if x.shape != (n,):
        raise ValueError("x0 must have shape (A.n_local,) = (%d,)" % n)

    info = _native.Info()
    err = ctypes.create_string_buffer(512)
    rc = lib.parcel_py_solve(ctypes.byref(o), *_matrix_args(A),
                             bb.ctypes.data, x.ctypes.data, ctypes.byref(info), err, 512)
    if rc != _native.OK:
        _raise(rc, err)
    return x, SolveInfo(converged=(info.reason == 1), reason=_REASONS.get(info.reason, str(info.reason)),
                        iterations=int(info.iterations), residual=float(info.residual),
                        true_residual=float(info.true_residual), reason_code=int(info.reason))


def eigs(A, *, method="ks_arnoldi", tol=None, maxiter=None,
         precond=None, precond_options=None, precond_spectrum=False,
         m=None, k_keep=None, nev=None, max_restart=None, sigma=None, v0=None,
         orthogonalization=None,
         inner_method=None, inner_tol=None, inner_maxiter=None, inner_restart=None,
         inner_precond=None, inner_precond_options=None,
         verbose=False):
    """固有値問題 Av = λv を解き, (values, vectors, EigInfo) を返す。

    values は複素数の配列 (power / inverse_power は 1 個, ks_arnoldi は k_keep 個),
    vectors は形 (A.n_local, 個数) の配列で, 各列が担当行の部分。ks_arnoldi の固有値は絶対値の降順
    (shift-invert では sigma に近い順) で, 複素共役対の固有ベクトルは実部と虚部が連続する 2 列に入る。
    全ランクが同じ引数で呼ぶ (集団操作)。未指定 (None) の設定はライブラリの既定値になる。

    method: "power" (絶対値最大), "inverse_power" (絶対値最小), "ks_arnoldi" (絶対値の大きい側から複数)
    precond_spectrum: True で A K^-1 の固有値を求める (K は precond の前処理行列)
    m, k_keep, nev, max_restart, sigma (shift-invert), v0 (初期ベクトル): ks_arnoldi のみ
    orthogonalization ("cgs" / "mgs"。どちらも 2 回適用 = CGS2 / MGS2): ks_arnoldi のみ
    inner_*: 内部の線形ソルバ (inverse_power, ks_arnoldi の shift-invert)。inner_method は "pgmres" / "pcg"
    """
    method = str(method).lower()
    if method not in _native.EIGEN:
        raise ValueError("method must be one of %s (got %r)" % (list(_native.EIGEN), method))
    ks_args = (m, k_keep, nev, max_restart, sigma, v0, orthogonalization)
    if method != "ks_arnoldi" and any(v is not None for v in ks_args):
        raise ValueError("m / k_keep / nev / max_restart / sigma / v0 / orthogonalization apply only to method='ks_arnoldi'")
    given = [v is not None for v in (m, k_keep, nev)]
    if any(given) and not all(given):
        raise ValueError("m, k_keep and nev must be given together")

    lib = _native.lib()
    o = _native.Options.unset()
    o.solver = _native.SOLVERS[method]
    o.verbose = 1 if verbose else 0
    o.tol = _pos_float("tol", tol)
    o.maxiter = _pos_int("maxiter", maxiter)
    _apply_precond(o, precond, precond_options)
    o.precond_spectrum = 1 if precond_spectrum else 0
    o.ks_m, o.ks_k_keep, o.ks_nev = _pos_int("m", m), _pos_int("k_keep", k_keep), _pos_int("nev", nev)
    o.ks_max_restart = _pos_int("max_restart", max_restart)
    o.ks_orthogonalization = _choice("orthogonalization", orthogonalization, _KS_ORTH)
    if sigma is not None:
        o.use_shift, o.sigma = 1, float(sigma)
    o.inner_method = _choice("inner_method", inner_method, _INNER)
    o.inner_tol = _pos_float("inner_tol", inner_tol)
    o.inner_maxiter = _pos_int("inner_maxiter", inner_maxiter)
    o.inner_restart = _pos_int("inner_restart", inner_restart)
    _apply_precond(o, inner_precond, inner_precond_options, inner=True)

    k = int(lib.parcel_py_eig_count(ctypes.byref(o)))
    if k < 1:
        raise ParcelError("could not determine the number of eigenvalues", k)
    n = A.n_local
    evals = np.zeros(2 * k)
    evecs = np.zeros(n * k)
    if v0 is not None:
        vv = np.asarray(v0, dtype=np.float64)
        if vv.shape != (n,):
            raise ValueError("v0 must have shape (A.n_local,) = (%d,)" % n)
        evecs[:n] = vv
        o.use_v0 = 1

    info = _native.Info()
    err = ctypes.create_string_buffer(512)
    rc = lib.parcel_py_eigs(ctypes.byref(o), *_matrix_args(A),
                            evals.ctypes.data, evecs.ctypes.data, ctypes.byref(info), err, 512)
    if rc != _native.OK:
        _raise(rc, err)
    values = evals[0::2] + 1j * evals[1::2]
    vectors = evecs.reshape(k, n).T.copy()   # 列 j が j 番目の固有ベクトル
    return values, vectors, EigInfo(
        converged=(info.reason == 1), reason=_REASONS.get(info.reason, str(info.reason)),
        iterations=int(info.iterations), residual=float(info.residual),
        true_residual=float(info.true_residual), reason_code=int(info.reason),
        n_converged=int(info.ks_nconv), inner_failures=int(info.linsolve_failures))
