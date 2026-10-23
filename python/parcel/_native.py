# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""ネイティブ層 libparcel_py.so の読み込み (ctypes)。

C / Fortran の API (parcel_sycl_c.h) とは独立した, Python 層専用のネイティブ層を使う
(宣言は python/src/parcel_py.h)。
"""
import atexit
import ctypes
import os

ABI_VERSION = 3

# ソルバの番号 (parcel_py.h)
SOLVERS = {
    "pcg": 0, "pbicgstab": 1, "pgmres": 2, "pcagmres": 3,
    "power": 10, "inverse_power": 11, "ks_arnoldi": 12,
}
LINEAR = ("pcg", "pbicgstab", "pgmres", "pcagmres")
EIGEN = ("power", "inverse_power", "ks_arnoldi")

OK, ERR_ARG, ERR_NOT_BUILT, ERR_OVERFLOW, ERR_MPI, ERR_RUNTIME = 0, -1, -2, -3, -4, -5

_i32, _i64, _f64 = ctypes.c_int32, ctypes.c_int64, ctypes.c_double


class Options(ctypes.Structure):
    _fields_ = [
        ("abi_version", _i32), ("solver", _i32), ("verbose", _i32),
        ("tol", _f64), ("maxiter", _i64),
        ("precond", _i32), ("jacobi_iter", _i64), ("blocks", _i64),
        ("scale_matrix", _i32), ("scale_vector", _i32),
        ("restart", _i64), ("orthogonalization", _i32),
        ("s", _i64), ("t", _i64), ("qr", _i32), ("q_mode", _i32),
        ("precond_spectrum", _i32),
        ("ks_m", _i64), ("ks_k_keep", _i64), ("ks_nev", _i64), ("ks_max_restart", _i64),
        ("use_shift", _i32), ("sigma", _f64), ("use_v0", _i32),
        ("inner_method", _i32), ("inner_tol", _f64), ("inner_maxiter", _i64),
        ("inner_restart", _i64), ("inner_precond", _i32), ("inner_jacobi_iter", _i64),
        ("inner_blocks", _i64),
        ("ks_orthogonalization", _i32),
    ]

    @classmethod
    def unset(cls):
        """すべての項目を「未指定」にした構造体 (ライブラリの既定値を使う)。"""
        o = cls()
        nan = float("nan")
        for name, typ in cls._fields_:
            setattr(o, name, nan if typ is _f64 else -1)
        o.abi_version = ABI_VERSION
        o.verbose = 0
        o.precond_spectrum = 0
        o.use_shift = 0
        o.use_v0 = 0
        return o


class Info(ctypes.Structure):
    _fields_ = [
        ("reason", _i32), ("iterations", _i64), ("residual", _f64), ("true_residual", _f64),
        ("ks_nconv", _i64), ("linsolve_failures", _i64), ("n_eig", _i64),
    ]


_lib = None


def _find_library():
    env = os.environ.get("PARCEL_PY_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "libparcel_py.so")


def lib():
    """ネイティブ層を読み込み, MPI を初期化して返す (初回だけ)。"""
    global _lib
    if _lib is not None:
        return _lib
    path = _find_library()
    if not os.path.exists(path):
        raise ImportError(
            "parcel: native library not found: %s "
            "(build it with python/build_<environment>.sh, or set PARCEL_PY_LIB)" % path)
    # AdaptiveCpp はキューを作るたびに使わないバックエンドの警告を出す (1 回の呼び出しごとに出る)。
    #   利用者が指定していなければ警告を止める (ACPP_DEBUG_LEVEL=1 以上で再び出る)。
    os.environ.setdefault("ACPP_DEBUG_LEVEL", "0")
    # Open MPI は自身のプラグインから libmpi の記号を探すので RTLD_GLOBAL で読み込む
    L = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    P = ctypes.POINTER
    L.parcel_py_abi_version.restype = ctypes.c_int
    L.parcel_py_version_string.restype = ctypes.c_char_p
    L.parcel_py_has_mpi.restype = ctypes.c_int
    L.parcel_py_mpi_init.restype = ctypes.c_int
    L.parcel_py_mpi_finalize.restype = None
    L.parcel_py_comm_rank.argtypes = [ctypes.c_void_p]
    L.parcel_py_comm_size.argtypes = [ctypes.c_void_p]
    L.parcel_py_allreduce.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    L.parcel_py_allreduce.restype = ctypes.c_double
    L.parcel_py_eig_count.argtypes = [P(Options)]
    L.parcel_py_eig_count.restype = _i64
    L.parcel_py_build_info.argtypes = [ctypes.c_char_p, ctypes.c_int]
    L.parcel_py_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    arr_args = [P(Options), ctypes.c_void_p, _i64, _i64, _i64, _i64,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, P(Info), ctypes.c_char_p, ctypes.c_int]
    L.parcel_py_solve.argtypes = arr_args
    L.parcel_py_eigs.argtypes = arr_args
    v = L.parcel_py_abi_version()
    if v != ABI_VERSION:
        raise ImportError("parcel: native library ABI %d does not match the Python package (%d); "
                          "rebuild the native library" % (v, ABI_VERSION))
    # MPI なしのビルド (-DPARCEL_MPI=0) では mpi_init は何もせず 0 を返す
    rc = L.parcel_py_mpi_init()
    if rc < 0:
        raise RuntimeError("parcel: MPI initialization failed")
    if rc == 1:
        # このパッケージが MPI を初期化したときだけ終了時に MPI_Finalize する
        # (mpi4py などが初期化していればそちらに任せる)
        atexit.register(L.parcel_py_mpi_finalize)
    _lib = L
    return _lib


def has_mpi():
    """ネイティブ層を MPI ありでビルドしたら True, MPI なし (1 プロセス版) なら False。"""
    return bool(lib().parcel_py_has_mpi())


def library_version():
    """ネイティブ層をビルドした PARCEL の版 ("1.2.0" の形。Include/parcel_version.hpp の値)。
    parcel.__version__ (このパッケージの版) と同じになる (tests/regression/sample_py.py の version で確かめる)。"""
    return lib().parcel_py_version_string().decode()


def device_name():
    """このランクで PARCEL が使うデバイスの名前 (既定のセレクタで選ばれるもの)。"""
    buf = ctypes.create_string_buffer(256)
    lib().parcel_py_device_name(buf, 256)
    return buf.value.decode(errors="replace")


def build_info():
    """ネイティブ層のビルドの情報 (整数型の幅, 精度, ターゲットとカーネル最適化のフラグ)。"""
    buf = ctypes.create_string_buffer(256)
    lib().parcel_py_build_info(buf, 256)
    info = buf.value.decode()
    try:
        from . import _build_config as c   # pip install で入れたときだけある
        info += " target=%s kernel_flags=%s" % (c.TARGET, c.KERNEL_FLAGS)
    except ImportError:
        pass
    return info
