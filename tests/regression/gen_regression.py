#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""回帰試験のジョブスクリプトの生成。

  python3 tests/regression/gen_regression.py [環境 ...]
  python3 tests/regression/gen_regression.py [--plat <環境,...>] [--mpi yes|no|both] [--lang cpp,c,fortran,python]

tests/regression/ に次を書き出す (環境を指定しなければ tests/site_config.py の REGRESSION の全環境)。
MPI あり / なし (--mpi) と言語 (--lang: cpp = 検証サンプルの全精度、c / fortran / python = C / Fortran / Python の API の試験) を
任意に組み合わせて選べる。選んだときは選んだ組の生成物だけを作り直し、他のスクリプトは残す。
  sub_build_<plat>.sh                          精度ごとの実行ファイル bin/<plat>/verify_<精度> と
                                               bin/<plat>/ddcheck を作るビルドジョブ
  sub_regression_<plat>_<精度>_<分割>_np<N>.sh 試験ジョブ (1 ジョブ = 1 精度 x ソルバの組 x 並列数)
  sub_regression_<plat>_ddcheck_np<N>.sh       double-double の劣化検出
MPI なしの構成 (site_config.py の REGRESSION_NOMPI) は環境名 <plat>_nompi で同じものを作る
(並列数は 1 だけ、ddcheck も np=1 だけ)。

検証サンプル (tests/verify/parcel_krylov_sample_verify.cpp) はソルバ・行列・前処理・問題サイズ・
反復数・検証の有無を実行時オプションで切り替えられるので、ビルドは「精度構成 x 環境」ごとに 1 回で済む。
環境ごとの設定 (キュー・モジュール・コンパイルフラグ) は tests/site_config.py にある。
"""
import glob
import os
import stat
import sys

sys.dont_write_bytecode = True       # パッケージの中に __pycache__ を作らない
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from site_config import REGRESSION as PLATFORMS, ENV_SETUP, PS_LAPACKE, PROJECT, \
    S86_CUDA_MODULE, S86_CUDA_ROOT  # noqa: E402
try:
    from site_config import REGRESSION_NOMPI, NOMPI_CXX  # noqa: E402
except ImportError:                  # MPI なしの構成を書いていない site_config.py
    REGRESSION_NOMPI, NOMPI_CXX = [], "acpp"
try:
    from site_config import REGRESSION_LANG, LANG_MPI_CC, LANG_MPI_FC, LANG_NOMPI_CC, LANG_NOMPI_FC  # noqa: E402
except ImportError:                  # C / Fortran の試験を書いていない site_config.py
    REGRESSION_LANG, LANG_MPI_CC, LANG_MPI_FC = [], {"default": "mpicc"}, {"default": "mpif90"}
    LANG_NOMPI_CC, LANG_NOMPI_FC = "gcc", "gfortran"
try:
    from site_config import REGRESSION_LANG_PY  # noqa: E402
except ImportError:                  # Python の試験を書いていない site_config.py
    REGRESSION_LANG_PY = []
from check_version import version_string  # noqa: E402

# 版 (Include/parcel_version.hpp)。試験に PARCEL_EXPECT_VERSION として渡し、検証サンプル ([VERSION] の行) と
#   C / Fortran / Python の試験 (version の項目) が実行時の版と照合する
PARCEL_VERSION = version_string()

# 生成したスクリプトはパッケージのルートから投入する (WORKDIR がルートになる)。
VERIFY_SRC = "tests/verify/parcel_krylov_sample_verify.cpp"
DDCHECK_SRC = "tests/regression/ddcheck/ddcheck.cpp"
SCRIPT_DIR = "tests/regression"

NX = 8                   # n = 512。全ケースで密行列の参照解を作れる大きさ
NPROC_LIST = [1, 2, 4]


# ---------------------------------------------------------------- MPI なしの構成
#   -DPARCEL_MPI=0 でビルドした実行ファイルを、mpirun / mpiexec を通さずに 1 プロセスで直接起動する。
#   REGRESSION_NOMPI に挙げた環境 <plat> から <plat>_nompi を派生させる。変えるのは次だけで、
#   キュー・資源 (np=1 と同じ)・ターゲット・精度・ケース・判定は元の環境と同じ。
#     ビルド : ENV_SETUP の CXX を NOMPI_CXX (acpp) に替え、-DPARCEL_MPI=0 を付ける (extra に足す)
#     実行   : np=1 だけ。起動コマンドは無し。GPU は割り当てスクリプトを通さず、MPI あり np=1 の
#              ランク 0 と同じ装置 (CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES = 0) を選ぶ。
#              ランクの NUMA 固定 (pin) は MPI の機能なので使わない。
#   結果は MPI あり np=1 と一致するはずなので、compare_nompi.py で突き合わせる。
NOMPI_SUFFIX = "_nompi"


def nompi_cfg(base):
    """MPI ありの環境 base から MPI なしの設定を作る"""
    cfg = dict(PLATFORMS[base])
    cfg["extra"] = (cfg["extra"] + " -DPARCEL_MPI=0").strip()
    cfg.update(nompi=True, base=base)
    for k in ("pin", "mpi_impl"):
        cfg.pop(k, None)
    return cfg


for _base in REGRESSION_NOMPI:
    if _base not in PLATFORMS:
        sys.exit(f"REGRESSION_NOMPI の {_base} が REGRESSION に無い")
NOMPI_PLATFORMS = {b + NOMPI_SUFFIX: nompi_cfg(b) for b in REGRESSION_NOMPI}
ALL_PLATFORMS = {**PLATFORMS, **NOMPI_PLATFORMS}


def nprocs_of(cfg):
    """試験する並列数。MPI なしは 1 プロセスだけ"""
    return [1] if cfg.get("nompi") else NPROC_LIST


def nps_text(cfg, nps):
    """ヘッダの「MPI = ...」の値"""
    if cfg.get("nompi"):
        return "なし (-DPARCEL_MPI=0。1 プロセスを mpirun なしで直接起動)"
    return nps


def env_setup_for(cfg, env):
    """ジョブの環境設定。MPI なしはコンパイラを MPI のラッパから NOMPI_CXX に替える"""
    if not cfg.get("nompi"):
        return env
    lines = env.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("CXX=")]
    if len(idx) != 1:
        sys.exit(f"ENV_SETUP[{cfg['env']!r}] に CXX= の行がちょうど 1 つ必要 ({len(idx)} 行)")
    lines[idx[0]] = ("# MPI なしのビルド (-DPARCEL_MPI=0): MPI のラッパを通さず直接コンパイルする\n"
                     f"CXX={NOMPI_CXX}")
    return "\n".join(lines)


def launch_for(cfg, np_):
    """np に合わせた起動コマンドを返す。

    Open MPI の既定は「1 ランク = 1 コア」なので、ハイブリッド並列では PE=<スレッド数> を
    明示しないとスレッドが 1 コアに詰め込まれる。np ごとにジョブを分けて
    ncpus = スレッド数 x np を確保してあるので、ppr:<np>:node:PE=<スレッド数> がちょうど収まる。
    富士通の mpiexec は --mpi / --omp で配置が決まり、Intel MPI は --map-by を解釈しない
    (I_MPI_PIN_DOMAIN を環境設定側で与える) ので、どちらも何も付けない。
    """
    if cfg.get("nompi"):
        return ""                        # MPI なし: 実行ファイルを直接起動する
    base = cfg["launch"]
    if base.startswith("mpiexec"):
        return base
    if cfg.get("mpi_impl") == "intel":
        return base
    if cfg.get("map_by") is False:
        return base                      # 配置を launch に書き切っている環境 (プロセスをコアに固定できない macOS など)
    omp = cfg["omp"]
    # 1 ノードに載せるランク数。1 ランク 1 チャンクで 1 ノードの装置数 (node_chunks) が決まっている環境 (PS-B) で
    #   np がそれを超えるときは、1 ノードに node_chunks ずつ (ceil(np / node_chunks) ノード。place_for の place[1] で取る)
    ppn = min(np_, cfg["node_chunks"]) if cfg.get("node_chunks") else np_
    opt = f" --map-by ppr:{ppn}:node:PE={omp} --bind-to core"
    # GPU の割り当てスクリプトはランチャの後ろに来るので、その前に挿入する
    if "${WORKDIR}" in base:
        i = base.index(" ${WORKDIR}")
        return base[:i] + opt + base[i:]
    return base + opt


def select_for(cfg, np_):
    """np に合わせた資源指定を返す。

    資源指定・起動コマンド・実際の np の 3 つが揃っていないとランクが正しく配置されないので、
    per_rank (1 ランクあたりの資源) を np 倍して組み立てる。
      chunk=False : 1 チャンクに np ランクを載せる
      chunk=True  : 1 ランク 1 チャンクを np 個取る
      node_ncpus  : np によらずノード全体を取る (ランクを NUMA 単位に固定するため)
    """
    per = cfg.get("per_rank")
    if cfg["sched"] == "tcs":
        if per is None:
            return cfg["select"]
        return per["tcs"].format(np=np_)
    if per is None:
        return cfg["select"]
    omp = per["ompthreads"]
    if per.get("node_ncpus"):
        return f"select=1:ncpus={per['node_ncpus']}:mpiprocs={np_}:ompthreads={omp}"
    if per.get("chunk"):
        t = f"select={np_}:ncpus={per['ncpus']}:mpiprocs=1:ompthreads={omp}"
        if per.get("ngpus"):
            t += f":ngpus={per['ngpus']}"
    else:
        t = f"select=1:ncpus={per['ncpus'] * np_}:mpiprocs={np_}:ompthreads={omp}"
        if per.get("ngpus"):
            t += f":ngpus={per['ngpus'] * np_}"
    return t


# ---------------------------------------------------------------- 精度構成 (ビルド単位)
#
#   tol     収束判定の閾値 (外側と、inverse_power / shift-invert の内部ソルバ)
#   imax    内部ソルバの反復上限
#   errmax  参照解との誤差の合格閾値 (power は収束が遅いので errmax_power)
#   tol_solver / errmax_solver  ソルバ (と行列) ごとの例外。キーはケース名の前方一致
#
#   閾値はその精度で到達できる相対残差に合わせる。
#   - dataF16 はベクトルも fp16 で保存するため、残差が 1e-2 を切ると丸めが残差と同程度になり
#     漸化式の残差が真の残差から離れる。到達できるのは 1e-2 程度。
#   - 前処理だけを低精度にする構成 (precon_*) は本体が倍精度なので double と同じ閾値を使う。
#     例外は、fp16 演算の前処理で pbicgstab が残差 ~1e-6 を下回ると NaN で破綻すること、
#     fp32 演算の前処理で ks_arnoldi の残差の床が 1e-9 に届かないこと、
#     前処理付き固有値の参照解 A M^-1 自体が fp16 の丸めを受けること (errmax は fp16 の eps に対応する 1e-3)。
#   - 単精度の前処理付き power は A M^-1 の丸めがそのまま固有値の誤差になる (5e-4)。
PRECISIONS = {
    "double":    dict(flags="",                                    tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4"),
    "calcF16":   dict(flags="-Dparcel_solver_calc_type=parcel_fp16",
                                                                 tol="1.0e-3", imax=200,  errmax="2.0e-2", errmax_power="2.0e-2"),
    "dataF16":   dict(flags="-Dparcel_solver_data_type=parcel_fp16 "
                            "-Dparcel_solver_calc_type=parcel_fp16",
                                                                 tol="1.0e-2", imax=200,  errmax="5.0e-2", errmax_power="5.0e-2"),
    # 前処理を低精度にする 6 構成: 演算のみ (calc) / データのみ (data) / 両方 (all)
    "precon_calcF16": dict(flags="-Dparcel_solver_calc_precon_type=parcel_fp16",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4",
                           tol_solver={"pbicgstab": "1.0e-5", "ks_arnoldi": "1.0e-3"},
                           # 移流拡散 (非対称で条件数が悪い) の閾値。fp16 前処理の丸めがポアソンより 1 桁大きく出る。
                           #   既定のケース一覧では double 以外は行列がポアソンだけなので、移流拡散を加えるときに使う
                           errmax_solver={"pbicgstab_convdiff": "1.0e-2",
                                          "ks_arnoldi_convdiff2": "1.0e-2",
                                          "pbicgstab": "1.0e-3", "ks_arnoldi": "1.0e-3",
                                          "power": "1.0e-3"}),
    "precon_dataF16": dict(flags="-Dparcel_solver_data_precon_mat_type=parcel_fp16 "
                                 "-Dparcel_solver_data_precon_vec_type=parcel_fp16",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4",
                           errmax_solver={"ks_arnoldi": "1.0e-3", "power": "1.0e-3"}),
    "precon_allF16":  dict(flags="-Dparcel_solver_calc_precon_type=parcel_fp16 "
                                 "-Dparcel_solver_data_precon_mat_type=parcel_fp16 "
                                 "-Dparcel_solver_data_precon_vec_type=parcel_fp16",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4",
                           tol_solver={"pbicgstab": "1.0e-5", "ks_arnoldi": "1.0e-3"},
                           errmax_solver={"pbicgstab_convdiff": "1.0e-2",
                                          "ks_arnoldi_convdiff2": "1.0e-2",
                                          "pbicgstab": "1.0e-3", "ks_arnoldi": "1.0e-3",
                                          "power": "1.0e-3"}),
    "calcF":     dict(flags="-Dparcel_solver_calc_type=parcel_float",
                                                                 tol="1.0e-6", imax=500,  errmax="1.0e-4", errmax_power="1.0e-4",
                           errmax_solver={"power_poisson_p4_JACOBI": "5.0e-4",
                                          "power_poisson_p2_BLOCKJACOBI": "5.0e-4"}),
    "dataF":     dict(flags="-Dparcel_solver_data_type=parcel_float "
                            "-Dparcel_solver_calc_type=parcel_float",
                                                                 tol="1.0e-6", imax=500,  errmax="1.0e-4", errmax_power="1.0e-4",
                           errmax_solver={"power_poisson_p4_JACOBI": "5.0e-4",
                                          "power_poisson_p2_BLOCKJACOBI": "5.0e-4"}),
    "precon_calcF":   dict(flags="-Dparcel_solver_calc_precon_type=parcel_float",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4",
                           tol_solver={"ks_arnoldi": "1.0e-6"}),
    "precon_dataF":   dict(flags="-Dparcel_solver_data_precon_mat_type=parcel_float "
                                 "-Dparcel_solver_data_precon_vec_type=parcel_float",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4"),
    "precon_allF":    dict(flags="-Dparcel_solver_calc_precon_type=parcel_float "
                                 "-Dparcel_solver_data_precon_mat_type=parcel_float "
                                 "-Dparcel_solver_data_precon_vec_type=parcel_float",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4",
                           tol_solver={"ks_arnoldi": "1.0e-6"}),
    # double-double は精度を上げるための指定で、収束判定の目標は double と同じにする。
    #   power の誤差はべき乗法の収束で決まるので double と同じ 1e-4。
    "calcDD":    dict(flags="-Dparcel_solver_calc_type=parcel_doubledouble",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4"),
    "dataDD":    dict(flags="-Dparcel_solver_data_type=parcel_doubledouble "
                            "-Dparcel_solver_calc_type=parcel_doubledouble",
                                                                 tol="1.0e-9", imax=5000, errmax="1.0e-6", errmax_power="1.0e-4"),
}

ALL7 = ["pcg", "pbicgstab", "pgmres", "pcagmres", "power", "inverse_power", "ks_arnoldi"]
# FP16 のデータ・演算精度 (calcF16 / dataF16) でも 7 ソルバすべてを試す。pcagmres は Q を陽に作る版 (explicit) だけで、
#   CholeskyQR 系 (implicit・モーメント法) は s, t によらず NaN・未収束になるので対象外 (下の CHOLQR_FP16)。
FP16_SOLVERS = list(ALL7)
PRECON_SOLVERS = ALL7

# ソルバごとの行列と反復上限
MATRICES = {"poisson": "", "convdiff": "0.05", "convdiff2": "2.0"}
SOLVERS = {
    "pcg":           dict(mats=["poisson"],                          it=10000),
    "pbicgstab":     dict(mats=["poisson", "convdiff"],              it=10000),
    # リスタート周期はライブラリの既定 (pgmres 30、pcagmres (s, t) = (8, 4)) で試す。
    #   検証サンプルは値 default のとき set_restart を呼ばない。精度ごとの例外は STM_OVERRIDES。
    "pgmres":        dict(mats=["poisson", "convdiff"],              it=10000,
                          extra="PARCEL_GMRES_RESTART=default"),
    "pcagmres":      dict(mats=["poisson", "convdiff"],              it=10000,
                          extra="PARCEL_CAGMRES_S=default"),
    # power は収束が線形で、ブロックヤコビ前処理付きでは A M^-1 の固有値の間隔が狭く遅い
    "power":         dict(mats=["poisson", "convdiff"],              it=5000),
    "inverse_power": dict(mats=["poisson"],                          it=200),
    # ks_arnoldi の反復の上限はリスタート回数 PARCEL_KS_RESTART で決まる (ライブラリの既定 200 で試す)
    "ks_arnoldi":    dict(mats=["poisson", "convdiff", "convdiff2"], it=None,
                          extra="PARCEL_KS_M=40 PARCEL_KS_K_KEEP=10 PARCEL_KS_NEV=4 PARCEL_KS_RESTART=default"),
}
PRECONS = [(0, "NONE"), (1, "POINTJACOBI"), (2, "BLOCKJACOBI"), (4, "JACOBI")]

# power x ブロックヤコビだけ反復の上限を 20000 にする。A M^-1 の固有値の間隔はブロックの区切り位置
#   (CPU ではランクあたりスレッド数で決まる) で大きく変わり、5000 回に届く分割がある。
POWER_BJ_MAX_ITER = 20000
def max_iter_of(sol, pv, info):
    if sol == "power" and pv == 2:
        return POWER_BJ_MAX_ITER
    return info["it"]

# ---------------------------------------------------------------- A M^-1 の固有値モード
#   inverse_power / ks_arnoldi shift-invert の set_precon_spectrum(true)。ソルバ名は擬似名で、
#   実際の PARCEL_SOLVER は "solver"。ks の shift-invert は PARCEL_KS_SHIFT で実行時に有効にする
#   (σ=0 で |λ| 最小側の 4 個)。精度ごとに "pspec" ジョブとして切り出す。
SOLVERS["invpow_pspec"] = dict(mats=["poisson"], it=200, solver="inverse_power",
                               extra="PARCEL_PRECON_SPECTRUM=1")
SOLVERS["kssi_pspec"] = dict(mats=["poisson", "convdiff"], it=None, solver="ks_arnoldi",
                             extra="PARCEL_PRECON_SPECTRUM=1 PARCEL_KS_SHIFT=0.0 "
                                   "PARCEL_KS_M=40 PARCEL_KS_K_KEEP=10 PARCEL_KS_NEV=4 PARCEL_KS_RESTART=default")
PSPEC = ["invpow_pspec", "kssi_pspec"]
PSPEC_PART = ("pspec", PSPEC, None)

# ---------------------------------------------------------------- Q を陽に作らない CA-GMRES
#   pcagmres の set_explicit_q(CAGMRES_Q_IMPLICIT) (QR は CholeskyQR)。
#   cholqr_moment は QR_CAGMRES_CHOLESKYQR (PARCEL_CHOLQR_MOMENT=1)。対称行列・前処理なし・
#   s が偶数のときだけ使えるので、前処理なしのポアソンだけを試す。
#   対象の精度は pcagmres と同じ。精度ごとに "cholqr" ジョブとして切り出す。
SOLVERS["cholqr_pcagmres"] = dict(mats=["poisson", "convdiff"], it=10000, solver="pcagmres",
                                  extra="PARCEL_CAGMRES_Q=implicit PARCEL_CAGMRES_S=default")
SOLVERS["cholqr_moment"] = dict(mats=["poisson"], it=10000, solver="pcagmres",
                                extra="PARCEL_CAGMRES_Q=implicit PARCEL_CHOLQR_MOMENT=1 PARCEL_CAGMRES_S=default")
CHOLQR = ["cholqr_pcagmres", "cholqr_moment"]
#   FP16 の演算・データ (calcF16 / dataF16) では CholeskyQR 系が s, t によらず NaN・未収束になるので対象外。
CHOLQR_FP16 = []
CHOLQR_PART = ("cholqr", CHOLQR, None)

# ---------------------------------------------------------------- ks_arnoldi の直交化
#   set_qr_algo(QR_CGS) (CGS2) と set_qr_algo(QR_MGS) (MGS2) を明示して試す (検証サンプルの PARCEL_KS_QR)。
#   既定の直交化は上の ks_arnoldi の全ケースで試しているので、ここは倍精度・ポアソン・前処理なしだけ。
#   ソルバ名は擬似名で、実際の PARCEL_SOLVER は "solver"。"ksqr" ジョブとして切り出す。
SOLVERS["ksqr_cgs"] = dict(mats=["poisson"], it=None, solver="ks_arnoldi",
                           extra="PARCEL_KS_M=40 PARCEL_KS_K_KEEP=10 PARCEL_KS_NEV=4 PARCEL_KS_RESTART=default "
                                 "PARCEL_KS_QR=cgs")
SOLVERS["ksqr_mgs"] = dict(mats=["poisson"], it=None, solver="ks_arnoldi",
                           extra="PARCEL_KS_M=40 PARCEL_KS_K_KEEP=10 PARCEL_KS_NEV=4 PARCEL_KS_RESTART=default "
                                 "PARCEL_KS_QR=mgs")
KSQR = ["ksqr_cgs", "ksqr_mgs"]
KSQR_PART = ("ksqr", KSQR, None)


def pspec_params(prec):
    """A M^-1 モードの (外側 tol, 内側 tol, 内側反復上限, errmax)

    内部 pgmres の相対残差は M^-1 の丸め・演算精度で床を持ち、床より厳しい内側 tol を
    指定すると毎回反復上限まで回るので、精度に合わせて緩める。
    """
    if prec in ("double", "calcDD", "dataDD"):
        return "1.0e-10", "1.0e-12", 5000, PRECISIONS[prec]["errmax"]
    if prec in ("calcF", "dataF", "precon_calcF", "precon_dataF", "precon_allF"):
        return "1.0e-6", "5.0e-6", 2000, "1.0e-5"
    if prec in ("precon_calcF16", "precon_dataF16", "precon_allF16"):
        return "1.0e-4", "3.0e-3", 500, "1.0e-2"
    if prec == "calcF16":
        return "1.0e-3", "3.0e-3", 500, "2.0e-2"
    if prec == "dataF16":
        return "1.0e-2", "2.0e-2", 500, "5.0e-2"
    raise KeyError(prec)


# ---------------------------------------------------------------- ジョブの分割
#   ks_arnoldi は他のソルバより 1 桁以上重いので行列ごとに 1 ジョブに切り離し、
#   残りは 2 つにまとめる。結果の出力先は分割によらず regression_results/<plat>/<精度>/。
SPLIT = [
    ("part1",        ["pcg", "pbicgstab", "pgmres"],            None),
    ("part2",        ["pcagmres", "power", "inverse_power"],    None),
    ("ks_poisson",   ["ks_arnoldi"], ["poisson"]),
    ("ks_convdiff",  ["ks_arnoldi"], ["convdiff"]),
    ("ks_convdiff2", ["ks_arnoldi"], ["convdiff2"]),
]
# 前処理を低精度にする 6 構成
MIXED_PRECON = {"precon_calcF16", "precon_dataF16", "precon_allF16",
                "precon_calcF", "precon_dataF", "precon_allF"}


# Intel MPI で、ノード内のランク k を NUMA ノード k に固定する。
#   I_MPI_PIN_DOMAIN に「NUMA ごとの先頭 <スレッド数> コア」の 16 進マスクを並べる。
#   マスクは計算ノードの /sys から作る (ハードコードしない)。
#   使い方: psa_numa_pin <ノードあたりのランク数> <スレッド数> [spread|pack]
PIN_SETUP = {
    "psa_numa": '\n# ランクを NUMA ノードに固定する (Intel MPI)。\n#   $1=ノードあたりのランク数 $2=スレッド数 $3=配置\n#     spread (既定): ランク k を NUMA k//ceil(ランク数/NUMA 数) に置く。回帰試験 (np=1/2/4) 用\n#     pack         : NUMA を先頭から順に埋める (1 NUMA に floor(最小 NUMA コア数/スレッド数) ランク)。\n#                    性能試験用\npsa_numa_pin () {\n  local M\n  M=$(python3 - "$1" "$2" "${3:-spread}" <<\'PYMASK\'\nimport sys, glob, re\nppn, per, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]\ndef expand(s):\n    out = []\n    for t in s.strip().split(","):\n        a, _, b = t.partition("-")\n        out += list(range(int(a), int(b or a) + 1))\n    return out\nnodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"),\n               key=lambda p: int(re.sub(r"\\D", "", p.rsplit("/", 1)[1])))\ncpus_of = [expand(open(p + "/cpulist").read()) for p in nodes]\nif mode == "pack":\n    rpn = min(len(c) for c in cpus_of) // per\n    if rpn < 1 or ppn > rpn * len(nodes):\n        sys.exit(f"ランク数 {ppn} が NUMA 収容数 {rpn} x {len(nodes)} を超える")\nelif mode == "spread":\n    rpn = -(-ppn // len(nodes))\nelse:\n    sys.exit(f"未知の配置 {mode}")\nms = []\nfor r in range(ppn):\n    i = r // rpn\n    cpus = cpus_of[i]\n    if len(cpus) < per * rpn:\n        sys.exit(f"{nodes[i]} のコア数 {len(cpus)} < {per} x {rpn}")\n    k = r % rpn\n    ms.append("0x" + format(sum(1 << c for c in cpus[k * per:(k + 1) * per]), "x"))\nprint("[" + ",".join(ms) + "]")\nPYMASK\n) || { echo "  NUMA マスクを作れない (ppn=$1, スレッド数=$2, 配置=${3:-spread})"; return 1; }\n  export I_MPI_PIN=1\n  export I_MPI_PIN_DOMAIN="${M}"\n  unset I_MPI_PIN_ORDER I_MPI_PIN_PROCESSOR_LIST\n  echo "  I_MPI_PIN_DOMAIN=${M}"\n}\n',
}


def pin_setup_for(cfg, ppn):
    """ランク固定のシェル断片 (str.format を通さずに埋め込む前提)。pin 指定の無い環境は空"""
    kind = cfg.get("pin")
    if not kind:
        return ""
    return PIN_SETUP[kind] + f"psa_numa_pin {ppn} {cfg['omp']} || exit 1\n"


# ランクごとに GPU を 1 台ずつ割り当てる。同時に走るジョブが書き換え合わないよう、
# スクリプトはジョブごとに別名にする。
GPU_BIND = """
# ランクごとに GPU を 1 台ずつ割り当てる。既定セレクタのままだと
# 全ランクが同じ GPU を掴みうる。
#   AMD では HIP_VISIBLE_DEVICES と ROCR_VISIBLE_DEVICES を両方設定すると多段に効いて
#   破綻するので、HIP 側だけを使い、ROCR 側は解除する。
cat > ${WORKDIR}/gpu_bind_{tag}.sh <<'BIND'
#!/bin/bash
R=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMIX_RANK:-0}}
export CUDA_VISIBLE_DEVICES=$R      # NVIDIA
export HIP_VISIBLE_DEVICES=$R       # AMD
unset ROCR_VISIBLE_DEVICES
exec "$@"
BIND
chmod +x ${WORKDIR}/gpu_bind_{tag}.sh
"""

def place_for(cfg, sel):
    """PBS の配置の行 (site_config.py の place)。チャンクの数が node_chunks 以下なら place[0]、超えれば place[1]

    PS-B (1 vnode = 1 APU、1 ノード = 4 vnode。1 ランク 1 チャンク) は ("pack", "exclhost"):
      np <= 4: pack (全チャンクを 1 ホストに。指定しないと B_S の既定 (free) で空いた vnode に散らばり、np=4 が 2〜3 ノードに
               またがる。2026-10-08 以前の生成器 (place の行なし) で起きた)
      np > 4 : exclhost (ホストを専有。チャンクは専有したホストの vnode を順に埋めるので ceil(np/4) ノードに 4 つずつ並ぶ。
               性能試験の装置数 8・16 で全件そうなっていた)。ランクの配置は launch_for が ppr:4:node にする
    1 チャンクで複数の GPU をまとめて取る指定 (select=1:ngpus=4) は、PS-B では 1 vnode (GPU 1 台) ぶんしか割り当てられない
    (PS の利用者ガイド) ので、1 ランク 1 チャンクのまま place で詰める。"""
    pl = cfg.get("place")
    if not pl or cfg["sched"] not in ("pbs", "pbs_miy", "pbs_ps"):
        return ""
    nchunk = int(sel.split(":")[0].split("=")[1])
    return f"#PBS -l place={pl[0] if nchunk <= cfg['node_chunks'] else pl[1]}\n"


# MPI なし (1 プロセス) の装置選択。MPI あり np=1 のランク 0 と同じ設定にする。
GPU_SELECT_NOMPI = """
# MPI なし (1 プロセス): MPI あり np=1 のランク 0 と同じ装置を選ぶ
export CUDA_VISIBLE_DEVICES=0       # NVIDIA
export HIP_VISIBLE_DEVICES=0        # AMD
unset ROCR_VISIBLE_DEVICES
"""


def gpu_bind_for(cfg, bindtag):
    if not cfg["gpu"]:
        return ""
    if cfg.get("nompi"):
        return GPU_SELECT_NOMPI
    return GPU_BIND.replace("{tag}", bindtag)


def parcel_load_of(cfg):
    """parcel_from="installed" の環境で、インストール済みの PARCEL を読み込んで PARCEL_PREFIX を決めるシェル断片。

    spack_parcel (parcel-sycl の構成。例 "~cuda~rocm"、"+cuda") に MPI の有無 (+mpi / ~mpi) を足して spack load する。
    """
    if parcel_from(cfg) != "installed":
        return ""
    if "spack_parcel" not in cfg:
        sys.exit("parcel_from='installed' には spack_parcel (parcel-sycl の構成) が要る")
    spec = cfg["spack_parcel"] + ("~mpi" if cfg.get("nompi") else "+mpi")
    return (f"# インストール済みの PARCEL (C / Fortran / Python の試験が使い、検証サンプルもこのヘッダで組む)\n"
            f"spack load parcel-sycl{spec} py-parcel-sycl{spec}\n"
            f"export PARCEL_PREFIX=$(spack location -i parcel-sycl{spec})\n")


def s86_env(cfg):
    """s86 形式 (sched="pbs") の環境設定。{cuda_mod} {cuda_env} {ld} {parcel_load} を埋める (GPU のときだけ CUDA を足す)"""
    cuda = cfg["gpu"]
    return ENV_SETUP[cfg.get("env", "s86")].format(
        cuda_mod=f"module load {S86_CUDA_MODULE}\n" if cuda else "",
        cuda_env=(f"export cuda_root={S86_CUDA_ROOT}\n"
                  "export ACPP_CUDA_PATH=$cuda_root\n") if cuda else "",
        ld="$cuda_root/lib64:" if cuda else "",
        parcel_load=parcel_load_of(cfg))


def header(plat, cfg, prec, part=None, np_=None, build=False):
    """ジョブのヘッダ。np_ を与えるとその並列数専用のジョブになる。"""
    name = f"{plat}_{prec}" + (f"_{part[0]}" if part else "")
    if np_ is not None:
        name += f"_np{np_}"
    sel = select_for(cfg, np_) if np_ is not None else cfg["select"]
    place = place_for(cfg, sel)
    nps = nps_text(cfg, str(np_) if np_ is not None else "/".join(map(str, NPROC_LIST)))
    tcs_mpi = f"proc={np_}" if np_ is not None else cfg.get("mpi", "")
    queue, walltime = cfg["queue"], cfg["walltime"]
    if build:
        queue = cfg.get("build_queue", queue)
        walltime = cfg.get("build_walltime", walltime)
    proj = PROJECT.get(cfg.get("project"), "")      # スケジューラの無い環境 (sched="local") は課金先が無い
    # s86 形式以外の環境設定は format しない (シェルの ${...} をそのまま書いている) ので、{parcel_load} だけを置き換える
    env_of = lambda key: env_setup_for(cfg, ENV_SETUP[key].replace("{parcel_load}", parcel_load_of(cfg)))
    desc = (f"# 回帰試験: {plat} / 精度 {prec}\n"
            f"#   NX={NX}, MPI = {nps}\n"
            f"#   ソルバ・行列・前処理は実行時オプションで切り替える"
            f"のでビルド不要 (bin/ の事前ビルドを使用)\n")
    if cfg["sched"] == "local":
        # スケジューラ無し: 資源の指定を書かず、tests/regression/run_local.sh が bash で流す
        return f"""#!/bin/bash
{desc}#   スケジューラの無い計算機: パッケージのルートで bash で流す (tests/regression/run_local.sh が順に流す)

{env_of(cfg["env"])}
"""
    if cfg["sched"] == "pbs_ps":
        return f"""#!/bin/bash
{desc}#PBS -N PARCEL_R_{name}
#PBS -P {proj}
#PBS -q {queue}
#PBS -l {sel}
{place}#PBS -l walltime={walltime}
#PBS -o regression_{name}.log
#PBS -j oe

{env_of(cfg["env"])}
"""
    if cfg["sched"] == "pbs_miy":
        return f"""#!/bin/bash
{desc}#PBS -q {queue}
#PBS -l {sel}
{place}#PBS -l walltime={walltime}
#PBS -W group_list={proj}
#PBS -N PARCEL_R_{name}
#PBS -j oe
#PBS -o regression_{name}.log

{env_of(cfg["env"])}
"""
    if cfg["sched"] == "pbs":
        env = env_setup_for(cfg, s86_env(cfg))
        return f"""#!/bin/bash
{desc}#PBS -N PARCEL_R_{name}
#PBS -P {proj}
#PBS -q {queue}
#PBS -l {sel}
{place}#PBS -l walltime={walltime}
#PBS -o regression_{name}.log
#PBS -e regression_{name}.err

{env}
"""
    return f"""#!/bin/bash
{desc}#PJM -L "rscgrp={queue}"
#PJM -L "{sel}"
#PJM -L "elapse={walltime}"
#PJM -g {proj}
#PJM --mpi {tcs_mpi}
#PJM --omp thread={cfg["thread"]}
#PJM -j
#PJM -N PARCEL_R_{name}
#PJM -o regression_{name}.log

{env_of(cfg["env"])}
"""


# 事前ビルドの実行ファイルを使う (ビルドは sub_build_<plat>.sh が行う)
BUILD_PREBUILT = """BIN=${{WORKDIR}}/bin/{plat}/verify_{prec}
if [ ! -x "${{BIN}}" ]; then
  echo "  BUILD_MISSING ${{BIN}}"
  echo "  先に {script_dir}/sub_build_{plat}.sh を実行して実行ファイルを作ること"
  exit 1
fi
echo "=== 事前ビルド済み {plat} / {prec} を使用 ==="
ls -l ${{BIN}}
"""

BODY = """
cd ${{WORKDIR}}
OUT=regression_results/{plat}/{prec}
mkdir -p ${{OUT}}
{gpu_bind}
@@PIN_SETUP@@
{build}

# $1=ケース名 $2=np $3...=実行時オプション (環境変数)
run_case () {{
  CASE=$1; NP=$2; shift 2
  D=${{OUT}}/${{CASE}}_np${{NP}}
  # 既に OK で終わっているケースは飛ばす (walltime 切れや再投入のときに続きから再開する)。
  #   最初からやり直すときは PARCEL_REG_FORCE=1 を付ける。
  if [ -z "${{PARCEL_REG_FORCE}}" ] && [ "$(cat ${{D}}/status.txt 2>/dev/null)" = "OK" ]; then
    echo "  [skip] ${{CASE}} np=${{NP}}"
    return
  fi
  # rank 0 が自分で書く結果ファイル (PARCEL_OUTPUT)。前回の残りを読まないよう消す。
  mkdir -p ${{D}}; rm -f ${{D}}/status.txt ${{D}}/parcel_result.txt
  export OMP_NUM_THREADS={omp}
  export PARALLEL=${{OMP_NUM_THREADS}}
  cd ${{D}}
  # 打ち切り時間。NX=8 は数秒で終わるが、ks_arnoldi と power は他より重いので長めにする。
  #   富士通の mpiexec は SIGTERM を無視することがあるので、-k で 30 秒後に SIGKILL を送る。
  case "${{CASE}}" in
    ks_arnoldi*|kssi_*|ksqr_*) CASE_TMO=300 ;;
    power*)      CASE_TMO=300 ;;
    *)           CASE_TMO=120 ;;
  esac
  T_START=$(date '+%F %T')
  timeout -k 30 ${{CASE_TMO}} env "$@" PARCEL_NX={nx} PARCEL_VERIFY=1 PARCEL_EXPECT_VERSION={version} {launch} ${{BIN}} > result.txt 2>&1 < /dev/null
  RC=$?
  # 実行したノード・時刻・終了コード (判定には使わない)
  echo "host=$(hostname) job=${{PBS_JOBID:-${{PJM_JOBID:-}}}} start=${{T_START}} end=$(date '+%F %T') rc=${{RC}} tmo=${{CASE_TMO}}" > run_info.txt
  # 失敗の直後は少し待つ (前のプロセスの後始末が終わる前に次の mpiexec を起動すると失敗する MPI がある)
  [ ${{RC}} -ne 0 ] && sleep 30
  cd ${{WORKDIR}}
  # 判定に使う出力。rank 0 が書く parcel_result.txt を優先し、無ければ標準出力。
  #   富士通の mpiexec では標準出力がジョブログへ出て result.txt が空になる。
  RES=${{D}}/parcel_result.txt
  [ -s "${{RES}}" ] || RES=${{D}}/result.txt
  # 線形ソルバの tol 未達を検出する。
  #   比べるのは停止条件に使う漸化式の残差 (反復履歴の `N : residual norm = ...`) で、
  #   `finish : residual true norm` (真の残差) ではない。低精度の構成では両者が離れ、
  #   CG は x の更新を pstep 反復ごとにまとめるので、真の残差は pstep 反復ごとにしか意味を持たない。
  #   解そのものの誤差は下の errmax で見る。
  #   inverse_power は内部ソルバのぶんも同じ行を出すので対象外 (外側は converged 行で判定する)。
  #   power の停止条件は固有値の相対変化なので対象外。
  NOTCONV=""
  case "${{CASE}}" in
    pcg*|pbicgstab*|pgmres*|pcagmres*|cholqr_*)
      TOL_C=""
      for A in "$@"; do
        case "$A" in PARCEL_TOL=*) TOL_C=${{A#PARCEL_TOL=}} ;; esac
      done
      RESID=$(grep -E "^ *[0-9]+ : residual norm" ${{RES}} 2>/dev/null | tail -1 | awk '{{print $NF}}')
      if [ -n "${{TOL_C}}" ] && [ -n "${{RESID}}" ] &&
         awk -v r="${{RESID}}" -v t="${{TOL_C}}" 'BEGIN{{exit !(r+0 > t+0)}}'; then
        NOTCONV=1
      fi
      ;;
  esac
  if [ ${{RC}} -eq 124 ]; then
    echo TIMEOUT > ${{D}}/status.txt
    # 打ち切ったケースのプロセスが残ると後続が巻き添えになるので消す
    pkill -u $(id -u) -f "$(basename ${{BIN}})" 2>/dev/null
    sleep 10
  elif [ ${{RC}} -ne 0 ]; then echo "RUN_FAIL_${{RC}}" > ${{D}}/status.txt
  elif grep -q "VERIFY_NONFINITE" ${{RES}}; then echo NONFINITE > ${{D}}/status.txt
  elif grep -Eqi "(^|[^a-z])-?nan([^a-z]|$)|(^|[^a-z])-?inf([^a-z]|$)" ${{RES}}; then
    echo NONFINITE > ${{D}}/status.txt
  elif grep -q "VERIFY_SKIPPED" ${{RES}}; then echo VERIFY_SKIPPED > ${{D}}/status.txt
  elif grep -q "converged = NO" ${{RES}}; then echo NOT_CONVERGED > ${{D}}/status.txt
  elif [ -n "${{NOTCONV}}" ]; then
    echo NOT_CONVERGED > ${{D}}/status.txt
    echo "    最終残差 ${{RESID}} > tol ${{TOL_C}}" >> ${{RES}}
  # 最後に解いたソルバ (固有値ソルバなら外側) の収束状態が DIVERGED_* なら未収束とする
  #   (power は誤差だけでは反復の上限に達したことを見分けられない)。
  elif grep -E "converged_reason = -?[0-9]+ \(" ${{RES}} | tail -1 | grep -q "(DIVERGED_"; then
    echo NOT_CONVERGED > ${{D}}/status.txt
    echo "    最終の収束状態: $(grep -E "converged_reason = -?[0-9]+ \(" ${{RES}} | tail -1)" >> ${{RES}}
  # 固有値ソルバの内部の線形ソルバが上限まで回った回数 (linsolve_failures) が 1 以上なら
  #   内部の停滞とする。外側が収束していても OK にしない。
  elif [ "$(grep -oE "linsolve_failures = [0-9]+" ${{RES}} | tail -1 | awk '{{print $3}}')" -gt 0 ] 2>/dev/null; then
    echo INNER_NOT_CONVERGED > ${{D}}/status.txt
  else
    # 参照解との誤差を閾値と比べる。
    #   線形ソルバ  : max|x_num - x_direct| / max|x_direct|
    #   固有値ソルバ: max relative eigenvalue err
    ERR=$(grep -m1 "x_direct| / max" ${{RES}} 2>/dev/null | awk '{{print $NF}}')
    [ -z "${{ERR}}" ] && ERR=$(grep -m1 "max relative eigenvalue err" ${{RES}} 2>/dev/null | awk '{{print $6}}')
    # 到達できる精度はソルバごとに違う (べき乗法は収束が線形で誤差が大きい)
    case "${{CASE}}" in
{errmax_cases}      *)      ERRMAX={errmax} ;;
    esac
    if [ -z "${{ERR}}" ] && ! grep -q "VERIFY" ${{RES}}; then
      # 検証の出力そのものが無い (出力を取り損ねている) ので OK にしない
      echo NO_VERIFY_OUTPUT > ${{D}}/status.txt
    elif [ -z "${{ERR}}" ]; then
      echo OK > ${{D}}/status.txt
    elif awk -v e="${{ERR}}" -v t="${{ERRMAX}}" 'BEGIN{{exit !(e+0 > t+0)}}'; then
      echo "ERROR_TOO_LARGE" > ${{D}}/status.txt
      echo "    誤差 ${{ERR}} > 閾値 ${{ERRMAX}}" >> ${{RES}}
    else
      echo OK > ${{D}}/status.txt
    fi
  fi
  echo "  [$(cat ${{D}}/status.txt)] ${{CASE}} np=${{NP}}  ($(cut -d' ' -f1,5 ${{D}}/run_info.txt 2>/dev/null))"
}}

"""


# ---------------------------------------------------------------- 低精度の個別の判定値
#   リスタート周期などはライブラリの既定で回し、低精度で残差が床に着いて破綻・停滞する組合せだけ、
#   判定値を実測の床の 10 倍前後に緩める。errmax (解・固有値の誤差の判定) も合わせて緩める。
#   gpu=True は GPU の環境だけに当てる (CPU は FP16 をエミュレートしていて床が低い)。
LOWPREC_OVERRIDES = [
    # FP16 の前処理 x 点ヤコビ: 漸化式の残差が 1e-6 の手前で止まる (同じ精度でもヤコビは 1e-10 まで届く)。
    dict(precs=("precon_calcF16", "precon_allF16"), sols=("pcg", "pgmres", "pcagmres", "cholqr_pcagmres"),
         precons=("POINTJACOBI",), gpu=None, tol="1.0e-5", errmax="1.0e-4"),
    #   inverse_power は内部 pgmres が 2e-6 台で止まるので、内部の判定値を緩める。
    dict(precs=("precon_calcF16", "precon_allF16"), sols=("inverse_power",),
         precons=("POINTJACOBI",), gpu=None, inner_tol="2.0e-5", errmax="1.0e-6"),
    # FP16 演算: calcF16 x pbicgstab (点ヤコビ・ブロックヤコビ) は 1e-3 台で破綻する (GPU。CPU でもブロックヤコビの
    #   分割によっては破綻する)。CPU と GPU で判定値をそろえる。
    dict(precs=("calcF16",), sols=("pbicgstab",), precons=("POINTJACOBI", "BLOCKJACOBI"),
         gpu=None, tol="3.0e-2", errmax="1.0e-1"),
    #   calcF16 x pcg x 点ヤコビは 1.4e-3 前後で NaN になることがある (GPU)。CPU にも同じ判定値を当てる。
    dict(precs=("calcF16",), sols=("pcg",), precons=("POINTJACOBI",),
         gpu=None, tol="1.0e-2", errmax="2.0e-2"),
    #   calcF16 / dataF16 x pcg x ブロックヤコビは 1e-2 台で NaN (GPU。CPU でも分割によっては NaN)。
    #   動作の確認に留まる判定値。
    dict(precs=("calcF16", "dataF16"), sols=("pcg",), precons=("BLOCKJACOBI",),
         gpu=None, tol="1.0e-1", errmax="2.0e-1"),
    # 固有値ソルバの内部の線形ソルバ (内部 pgmres) の判定値。内部が上限まで回ると INNER_NOT_CONVERGED になる。
    #   calcF16: 内部の残差が 4e-3〜9e-3 で止まる。
    dict(precs=("calcF16",), sols=("inverse_power", "invpow_pspec", "kssi_pspec"),
         precons=("NONE", "POINTJACOBI", "BLOCKJACOBI", "JACOBI"), gpu=None, inner_tol="5.0e-2"),
    #   FP16 の前処理 (演算) x A M^-1 モード: 内部の残差が 6e-4〜1.4e-2 で止まる (前処理なしは 1e-10 まで届く)。
    dict(precs=("precon_calcF16", "precon_allF16"), sols=("invpow_pspec", "kssi_pspec"),
         precons=("POINTJACOBI", "BLOCKJACOBI", "JACOBI"), gpu=None, inner_tol="5.0e-2"),
    #   FP16 の前処理 (データ) x ブロックヤコビ x inverse_power の A M^-1 モード: 内部の残差が 5e-3 程度で止まる。
    dict(precs=("precon_dataF16",), sols=("invpow_pspec",), precons=("BLOCKJACOBI",), gpu=None,
         inner_tol="3.0e-2"),
    # power: FP16 の前処理 x ブロックヤコビ・ヤコビは固有値の相対変化が 1e-9 に届かず反復の上限に達する。
    dict(precs=("precon_calcF16", "precon_allF16"), sols=("power",), precons=("BLOCKJACOBI",),
         gpu=None, tol="1.0e-5", errmax="2.0e-2"),
    dict(precs=("precon_calcF16", "precon_allF16"), sols=("power",), precons=("JACOBI",),
         gpu=None, tol="1.0e-5", errmax="3.0e-3"),
    # calcF16 x pcagmres: np=2 で残差が 2e-3〜7e-3 で停滞することがある (全前処理、全環境)。
    dict(precs=("calcF16",), sols=("pcagmres",), precons=("NONE", "POINTJACOBI", "BLOCKJACOBI", "JACOBI"),
         gpu=None, tol="1.0e-2"),
]


# ---------------------------------------------------------------- 精度ごとのリスタート周期
#   ライブラリの既定 (pgmres 30、pcagmres (8, 4)) から、NaN・未収束・反復数の大幅な増加を避けるために
#   変えるものだけを書く (階数落ちの表示が出るだけなら既定のまま)。
STM_OVERRIDES = [
    # calcF16 x pgmres: m = 30 だと残差が判定値 1e-3 のそばで停滞して反復数が大きく増える。
    dict(precs=("calcF16",), sols=("pgmres",), env={"PARCEL_GMRES_RESTART": "15"}),
]


def stm_env(prec, sol, env):
    """STM_OVERRIDES を環境変数のリストに当てる (同じ名前は置き換え、無ければ足す)"""
    for o in STM_OVERRIDES:
        if prec in o["precs"] and sol in o["sols"]:
            for k, v in o["env"].items():
                env = [e for e in env if not e.startswith(k + "=")] + [f"{k}={v}"]
    return env


def lowprec_overrides_for(prec, cfg):
    """(ソルバ, 前処理の名前) -> 上書きする dict。cfg が GPU かで gpu=True の行を選ぶ"""
    gpu = bool(cfg and cfg.get("gpu"))
    out = {}
    for o in LOWPREC_OVERRIDES:
        if prec not in o["precs"] or (o["gpu"] is True and not gpu):
            continue
        for sol in o["sols"]:
            for pn in o["precons"]:
                out[(sol, pn)] = o
    return out


def errmax_cases_for(prec, cfg=None):
    """errmax_solver をシェルの case 節に展開する。

    case 節は前方一致なので、行列まで指定した具体的なキー (pbicgstab_convdiff) を
    ソルバだけのキー (pbicgstab) より先に置く (キーの長い順に出す)。
    """
    ex = dict(PRECISIONS[prec].get("errmax_solver", {}))
    ex.setdefault("power", PRECISIONS[prec]["errmax_power"])
    for sol in PSPEC:
        ex[sol] = pspec_params(prec)[3]
    # 低精度の個別の判定値 (ケース名まで指定するのでソルバだけのキーより長く、先に出る)
    for (sol, pn), o in lowprec_overrides_for(prec, cfg).items():
        for mat in SOLVERS[sol]["mats"]:
            for pv, name in PRECONS:
                if name == pn and "errmax" in o:
                    ex[f"{sol}_{mat}_p{pv}_{pn}"] = o["errmax"]
    return "".join(f'      {sol}*) ERRMAX={v} ;;\n'
                   for sol, v in sorted(ex.items(), key=lambda kv: -len(kv[0])))


def cases_for(prec):
    """精度構成ごとに (ソルバ, 行列, ケース名, 実行時オプションのリスト) を返す"""
    out = []
    if prec == "double":
        solvers, precons, mats_all = ALL7, PRECONS, True
    elif prec in MIXED_PRECON:
        solvers, precons, mats_all = PRECON_SOLVERS, PRECONS, False
    elif prec in ("calcF16", "dataF16"):
        solvers, precons, mats_all = FP16_SOLVERS, PRECONS, False
    else:
        solvers, precons, mats_all = ALL7, PRECONS, False
    cholqr = CHOLQR_FP16 if prec in ("calcF16", "dataF16") else CHOLQR
    solvers = list(solvers) + PSPEC + (cholqr if "pcagmres" in solvers else []) + (KSQR if prec == "double" else [])
    for sol in solvers:
        info = SOLVERS[sol]
        # double 以外は行列をポアソンだけにする
        mats = info["mats"] if mats_all else ["poisson"]
        # ks_arnoldi x ブロックヤコビは重いので double だけで試す
        sol_precons = ([p for p in precons if p[0] != 2]
                       if sol.startswith("ks_arnoldi") and prec != "double" else precons)
        if sol == "cholqr_moment" or sol in KSQR:
            sol_precons = [p for p in precons if p[0] == 0]   # 前処理なし専用 (KSQR は前処理なしだけ試す)
        tol = PRECISIONS[prec].get("tol_solver", {}).get(sol, PRECISIONS[prec]["tol"])
        inner_tol, inner_max = PRECISIONS[prec]["tol"], PRECISIONS[prec]["imax"]
        if sol in PSPEC:
            tol, inner_tol, inner_max, _ = pspec_params(prec)
        for mat in mats:
            for pv, pn in sol_precons:
                env = [f"PARCEL_SOLVER={info.get('solver', sol)}",
                       f"PARCEL_MATRIX={'convdiff' if mat.startswith('convdiff') else 'poisson'}",
                       f"PARCEL_PRECON={pv}",
                       *([f"PARCEL_MAX_ITER={max_iter_of(sol, pv, info)}"] if info["it"] is not None else []),
                       f"PARCEL_TOL={tol}",
                       # 内部ソルバ (inverse_power / shift-invert) の閾値は外側と同じでよい
                       f"PARCEL_INNER_TOL={inner_tol}",
                       f"PARCEL_INNER_MAX={inner_max}"]
                if MATRICES[mat]:
                    env.append(f"PARCEL_CONVDIFF_C={MATRICES[mat]}")
                if "extra" in info:
                    env.extend(info["extra"].split())
                # 前処理付きの power / シフトなし ks_arnoldi は A M^-1 の固有値を試す。
                #   inverse_power の前処理付きのケースは内部ソルバの前処理として指定する
                #   (p0 のケースは内部ソルバがライブラリの既定 (ヤコビ) で解く)。
                if pv != 0 and sol in ("power", "ks_arnoldi"):
                    env.append("PARCEL_PRECON_SPECTRUM=1")
                if pv != 0 and sol == "inverse_power":
                    env.append(f"PARCEL_LINSOLVE_PRECON={pv}")
                env = stm_env(prec, sol, env)
                out.append((sol, mat, f"{sol}_{mat}_p{pv}_{pn}", env))
    return out


def cases_for_part(prec, part, cfg=None):
    """分割の定義でソルバ・行列を絞ったケースの一覧"""
    cases = cases_for(prec)
    # 低精度の個別の判定値を当てる (LOWPREC_OVERRIDES)
    ov = lowprec_overrides_for(prec, cfg)
    if ov:
        new = []
        for sol, mat, c, env in cases:
            o = ov.get((sol, c.rsplit("_", 1)[1]))
            if o:
                env = [(f"PARCEL_TOL={o['tol']}" if e.startswith("PARCEL_TOL=") and "tol" in o else
                        f"PARCEL_INNER_TOL={o['inner_tol']}" if e.startswith("PARCEL_INNER_TOL=") and "inner_tol" in o else e)
                       for e in env]
            new.append((sol, mat, c, env))
        cases = new
    if part is None:
        return [(c, e) for sol, _, c, e in cases if sol not in PSPEC and sol not in CHOLQR and sol not in KSQR]
    _, sols, mats = part
    return [(c, e) for sol, mat, c, e in cases
            if sol in sols and (mats is None or mat in mats)]


def lapack_of(cfg):
    """lapack=None は PS 用の指定 (nvhpc のヘッダ + OpenBLAS) に解決する"""
    return cfg["lapack"] if cfg["lapack"] is not None else PS_LAPACKE


def blocking_flags(cfg):
    """CPU では縮約 (dot / dddot / max) をブロック化する。

    平坦版の縮約は nd_range + barrier でツリー削減するが、AdaptiveCpp の OpenMP バックエンドでは
    barrier の固定費が大きい。ブロック化版は barrier を使わず部分和を work item ごとに持つ。
    GPU は平坦版が正しい形なので何も付けない。A64FX (_BLOCKING_LOOP_) は全カーネルがブロック化済み。
    分割数は実行時に omp_get_max_threads() から決まる (環境変数 PARCEL_NBLOCK で上書き可)。
    """
    if cfg.get("gpu"):
        return ""
    if "_BLOCKING_LOOP_" in cfg["extra"]:
        return ""
    return " -D_BLOCKING_LOOP_DOT_=1"


def spmv_flags(cfg):
    """SpMV と前処理のカーネル形状。

    CRS の内側ループを 4 -> 2 -> 1 の梯子状に分割すると、コンパイラが展開・先読みできる
    固定長の本体に全量が入る。AMD の GPU (MI300A) では SpMV・前処理とも速くなる。NVIDIA の GPU では
    V100 は平坦より遅く、A100・H100 は差が無かったので付けない (site_config.py で ladder=False にした環境)。
    CPU では中立なので付けない。A64FX は SpMV 側だけ有効にする (前処理側は遅くなる)。
    """
    if is_ody(cfg):
        return " -DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=0"
    if not cfg.get("gpu") or cfg.get("ladder") is False:
        return ""
    return " -DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1"


def is_ody(cfg):
    """A64FX (Wisteria-Odyssey) の環境か (ENV_SETUP の ody / odyspack)"""
    return cfg.get("env") in ("ody", "odyspack")


def target_of(cfg):
    return cfg["target"] + " " + cfg["extra"] + blocking_flags(cfg) + spmv_flags(cfg)


# 1 環境ぶんのビルドジョブ。
#   精度は互いに独立なので並列にビルドする (並列数は既定で精度の数、build_par で変えられる)。
#   コンパイルは MPI も GPU も使わないので、ログインノードで bash で実行してよい
#   (ヘッダは残してあるので qsub / pjsub でも動く)。
#   再実行可能: 出来ている実行ファイルは飛ばす。一部が落ちても再実行すれば残りだけ作る。
BUILD_JOB = """
WORKDIR=${{PBS_O_WORKDIR:-${{PJM_O_WORKDIR:-$(pwd)}}}}
cd ${{WORKDIR}}
if [ ! -f {verify_src} ]; then
  echo "パッケージのルートで実行すること ({verify_src} が無い: ${{WORKDIR}})"; exit 1
fi

BINDIR=${{WORKDIR}}/bin/{plat}
LOGDIR=${{WORKDIR}}/buildlog/{plat}
mkdir -p ${{BINDIR}} ${{LOGDIR}}

# 追加のコンパイルフラグ (既定は無し)。全精度と ddcheck に付く。
#   例: PARCEL_BUILD_EXTRA=-ffp-contract=off bash {script_dir}/sub_build_{plat}.sh
#   -ffp-contract=off は積和の FMA への融合を禁じ、ビルドし直しても結果のビット列が変わらないようにする。
#   bin/{plat}/build_extra.txt に記録し、違うフラグで作った実行ファイルが混ざらないよう止める。
EXTRA=${{PARCEL_BUILD_EXTRA:-}}
if [ -f ${{BINDIR}}/build_extra.txt ]; then
  PREV=$(cat ${{BINDIR}}/build_extra.txt)
elif ls ${{BINDIR}}/verify_* ${{BINDIR}}/ddcheck 2> /dev/null | grep -q .; then
  PREV=""
else
  PREV=${{EXTRA}}
fi
if [ "${{PREV}}" != "${{EXTRA}}" ]; then
  echo "bin/{plat} の実行ファイルは PARCEL_BUILD_EXTRA='${{PREV}}' で作られている (今回は '${{EXTRA}}')。"
  echo "bin/{plat} を空にしてから流すこと"; exit 1
fi
echo "${{EXTRA}}" > ${{BINDIR}}/build_extra.txt
[ -n "${{EXTRA}}" ] && echo "  追加フラグ: ${{EXTRA}}"

# 同時に走らせるコンパイルの数 (環境変数 PARCEL_BUILD_PAR で上書きできる。共有のログインノードでは小さめに)
NPAR=${{PARCEL_BUILD_PAR:-{npar}}}
build_one () {{   # $1=精度 $2...=精度フラグ
  P=$1; shift
  OUT=${{BINDIR}}/verify_${{P}}
  if [ -x "${{OUT}}" ]; then echo "  [skip] ${{P}} (既にある)"; return; fi
  T0=$(date +%s)
  ${{CXX}} -O3 {target} -DNX={nx} "$@" ${{EXTRA}} {lapack} \\
      -o ${{OUT}} {verify_src} ${{LINKOPT}} \\
      > ${{LOGDIR}}/${{P}}.log 2>&1
  if [ -x "${{OUT}}" ]; then
    echo "  [OK]   ${{P}}  $(( $(date +%s)-T0 )) 秒"
  else
    echo "  [FAIL] ${{P}}"; grep -E "error:" ${{LOGDIR}}/${{P}}.log | head -3 | sed 's/^/         /'
  fi
}}

# double-double が double に落ちていないかを見る検査プログラム。
#   calcDD と同じターゲット・フラグで作る。名前を verify_* にしないのは、
#   feed_regression.sh がビルドの完了を verify_* の本数で数えているため。
build_ddcheck () {{   # $@=calcDD の精度フラグ
  OUT=${{BINDIR}}/ddcheck
  if [ -x "${{OUT}}" ]; then echo "  [skip] ddcheck (既にある)"; return; fi
  T0=$(date +%s)
  ${{CXX}} -O3 {target} -DNX={nx} "$@" ${{EXTRA}} {lapack} \\
      -o ${{OUT}} {ddcheck_src} ${{LINKOPT}} \\
      > ${{LOGDIR}}/ddcheck.log 2>&1
  if [ -x "${{OUT}}" ]; then
    echo "  [OK]   ddcheck  $(( $(date +%s)-T0 )) 秒"
  else
    echo "  [FAIL] ddcheck"; grep -E "error:" ${{LOGDIR}}/ddcheck.log | head -3 | sed 's/^/         /'
  fi
}}

echo "=== {plat} の実行ファイルを作る (${{NPAR}} 並列) ==="
T_ALL=$(date +%s)
"""

# parcel_from="installed": 検証サンプル・ddcheck をインストール済みの PARCEL のヘッダで組む
BUILD_INSTALLED_HEADERS = """# PARCEL はインストール済みのもの ($PARCEL_PREFIX) を使う (site_config.py の parcel_from="installed")。
#   検証サンプルと ddcheck は Include/ を相対パスで読むので、試験のソース (tests/verify/、tests/regression/ddcheck/) を
#   buildlog/ の下に写し、その隣の Include を $PARCEL_PREFIX/include に向けて組む (パッケージの Include/ は読まない)。
if [ ! -f "${{PARCEL_PREFIX:-}}/include/parcel_sycl.hpp" ]; then
  echo "PARCEL_PREFIX (${{PARCEL_PREFIX:-未設定}}) に include/parcel_sycl.hpp が無い (環境設定で spack load などをすること)"; exit 1
fi
STAGE=${{LOGDIR}}/installed_src
rm -rf ${{STAGE}}; mkdir -p ${{STAGE}}/tests/regression
cp -r tests/verify ${{STAGE}}/tests/ && cp -r tests/regression/ddcheck ${{STAGE}}/tests/regression/ || exit 1
ln -s ${{PARCEL_PREFIX}}/include ${{STAGE}}/Include
VSRC=${{STAGE}}/{verify_src}
DSRC=${{STAGE}}/{ddcheck_src}
echo "${{PARCEL_PREFIX}}" > ${{BINDIR}}/parcel_prefix.txt
echo "  PARCEL のヘッダ: $(readlink -f ${{STAGE}}/Include) (インストール済み)"
"""


def build_job(plat, cfg):
    """1 環境ぶんのビルドジョブを返す"""
    npar = cfg.get("build_par", len(PRECISIONS))
    s = header(plat, cfg, "build", None, NPROC_LIST[-1], build=True)
    s = s.replace("PARCEL_R_" + plat + "_build_np" + str(NPROC_LIST[-1]),
                  "PARCEL_B_" + plat)
    s = s.replace(f"regression_{plat}_build_np{NPROC_LIST[-1]}",
                  f"build_{plat}")
    old = (f"# 回帰試験: {plat} / 精度 build\n"
           f"#   NX={NX}, MPI = {nps_text(cfg, NPROC_LIST[-1])}\n"
           f"#   ソルバ・行列・前処理は実行時オプションで切り替える"
           f"のでビルド不要 (bin/ の事前ビルドを使用)\n")
    assert s.count(old) == 1, plat
    where = ("計算ノードで実行する (pjsub で投入)" if cfg["sched"] == "tcs" and is_ody(cfg)
             else "ログインノードで bash で実行してよい (qsub でも動く)")
    s = s.replace(old,
                  f"# 回帰試験・性能試験の実行ファイルを作る: {plat}\n"
                  f"#   {len(PRECISIONS)} 精度ぶんを bin/{plat}/verify_<精度> に、dd の検査を bin/{plat}/ddcheck に作る。\n"
                  f"#   試験ジョブ (sub_regression_{plat}_*.sh) より先に済ませること。{where}。\n"
                  f"#   出来ている実行ファイルは飛ばすので、再実行すれば残りだけ作る。\n"
                  + (f"#   MPI なし: {cfg['base']} と同じフラグに -DPARCEL_MPI=0 を足し、MPI のラッパを通さずに組む。\n"
                     if cfg.get("nompi") else ""))
    s += BUILD_JOB.format(
        plat=plat, npar=npar, nx=NX, verify_src=VERIFY_SRC, ddcheck_src=DDCHECK_SRC,
        script_dir=SCRIPT_DIR, target=target_of(cfg), lapack=lapack_of(cfg))
    if parcel_from(cfg) == "installed":
        # 検証サンプルと ddcheck はライブラリを ../../Include (ddcheck は ../../../Include) の相対パスで読むので、
        #   試験のソースを作業域に写し、その隣の Include をインストールしたヘッダ ($PARCEL_PREFIX/include) に向けて組む
        for old, new in ((f"-o ${{OUT}} {VERIFY_SRC} ", "-o ${OUT} ${VSRC} "),
                         (f"-o ${{OUT}} {DDCHECK_SRC} ", "-o ${OUT} ${DSRC} ")):
            assert s.count(old) == 1, (plat, old)
            s = s.replace(old, new)
        old = f'echo "=== {plat} の実行ファイルを作る (${{NPAR}} 並列) ==="\n'
        assert s.count(old) == 1, plat
        s = s.replace(old, BUILD_INSTALLED_HEADERS.format(verify_src=VERIFY_SRC, ddcheck_src=DDCHECK_SRC) + old)
    for prec in PRECISIONS:
        s += ('while [ "$(jobs -rp | wc -l)" -ge ${NPAR} ]; do wait -n 2>/dev/null || sleep 2; done\n'
              f'build_one {prec} {PRECISIONS[prec]["flags"]} &\n')
    s += ('while [ "$(jobs -rp | wc -l)" -ge ${NPAR} ]; do wait -n 2>/dev/null || sleep 2; done\n'
          f'build_ddcheck {PRECISIONS[DDCHECK_PREC]["flags"]} &\n')
    s += 'wait\n'
    s += 'echo "  ビルド全体 $(( $(date +%s)-T_ALL )) 秒"\n'
    s += '\necho "=== ALL DONE ==="\n'
    return s


# ---------------------------------------------------------------- dd の劣化検出 (ddcheck)
#   -ffast-math などで double-double の誤差項が消えて dd が黙って double に落ちても、
#   calcDD / dataDD の解の誤差の閾値 (double 水準) では検出できない。ddcheck は
#   ソルバの dd 構成と同じ parcel::blas の経路に「厳密値が既知で double では表せない」入力を
#   与えて確かめる。np=1 (ランク内の縮約) と np=4 (MPI の dd 集約・袖通信) の 2 本。
DDCHECK_PREC = "calcDD"
DDCHECK_NP = [1, 4]
DDCHECK_TIMEOUT = 120

DDCHECK_BODY = """
cd ${{WORKDIR}}
BIN=${{WORKDIR}}/bin/{plat}/ddcheck
D=regression_results/{plat}/ddcheck/ddcheck_np{np}
mkdir -p ${{D}}; rm -f ${{D}}/status.txt ${{D}}/result.txt ${{D}}/ddcheck_result.txt
if [ ! -x "${{BIN}}" ]; then
  echo "  BUILD_MISSING ${{BIN}}"
  echo "  先に {script_dir}/sub_build_{plat}.sh を実行して bin/{plat}/ddcheck を作ること"
  echo BUILD_MISSING > ${{D}}/status.txt
  exit 1
fi
echo "=== ddcheck {plat} np={np} ==="
ls -l ${{BIN}}
NP={np}      # 起動コマンドが参照する
{gpu_bind}
@@PIN_SETUP@@
export OMP_NUM_THREADS={omp}
export PARALLEL=${{OMP_NUM_THREADS}}
cd ${{D}}
# rank 0 が ddcheck_result.txt (カレント) にも書く
timeout -k 30 {tmo} {launch} ${{BIN}} > result.txt 2>&1 < /dev/null
RC=$?
cd ${{WORKDIR}}
RES=${{D}}/ddcheck_result.txt
[ -s "${{RES}}" ] || RES=${{D}}/result.txt
if [ ${{RC}} -eq 124 ]; then echo TIMEOUT > ${{D}}/status.txt
elif [ ${{RC}} -eq 0 ] && grep -q "^\\[DDCHECK\\] RESULT PASS" ${{RES}}; then echo OK > ${{D}}/status.txt
elif grep -q "^\\[DDCHECK\\] RESULT FAIL" ${{RES}}; then echo DD_DEGRADED > ${{D}}/status.txt
elif [ ${{RC}} -ne 0 ]; then echo "RUN_FAIL_${{RC}}" > ${{D}}/status.txt
else echo NO_DDCHECK_OUTPUT > ${{D}}/status.txt
fi
grep "^\\[DDCHECK\\]" ${{RES}} 2>/dev/null | sed 's/^/    /'
echo "  [$(cat ${{D}}/status.txt)] ddcheck np={np}"

echo "=== ALL DONE ==="
"""


def gen_ddcheck(plat, cfg, np_):
    """dd の劣化検出のジョブ (1 環境・1 並列数) を返す"""
    s = header(plat, cfg, "ddcheck", None, np_)
    old = (f"# 回帰試験: {plat} / 精度 ddcheck\n"
           f"#   NX={NX}, MPI = {nps_text(cfg, np_)}\n"
           f"#   ソルバ・行列・前処理は実行時オプションで切り替える"
           f"のでビルド不要 (bin/ の事前ビルドを使用)\n")
    new = (f"# 回帰試験 ddcheck: {plat} / MPI = {nps_text(cfg, np_)}\n"
           f"#   double-double がデバイス側の BLAS (dot / axpy / gemv / SpMV) で dd の精度を\n"
           f"#   保っているか (double に落ちていないか) を確かめる。\n"
           f"#   bin/{plat}/ddcheck (sub_build_{plat}.sh が calcDD と同じフラグで作る) を使う。\n")
    assert s.count(old) == 1, (plat, s[:400])
    s = s.replace(old, new)
    bindtag = f"ddcheck_np{np_}"
    s += DDCHECK_BODY.format(
        plat=plat, np=np_, omp=cfg["omp"], tmo=DDCHECK_TIMEOUT, script_dir=SCRIPT_DIR,
        launch=launch_for(cfg, np_).replace("TAG", bindtag),
        gpu_bind=gpu_bind_for(cfg, bindtag))
    s = s.replace("@@PIN_SETUP@@", pin_setup_for(cfg, np_))
    return rank_out_fix(cfg, s)


def local_no_pkill(cfg, s, line, indent):
    """スケジューラの無い環境 (sched="local") では、打ち切ったときの pkill をしない。
    run_local.sh は同じ実行ファイルを使うジョブを並べて流すので、名前で探して止めると巻き込む。
    打ち切りは timeout (site_config.py の ENV_SETUP で用意する) が子に TERM / KILL を送る。"""
    if cfg.get("sched") != "local":
        return s
    assert s.count(line) == 1, line
    return s.replace(line, f"{indent}# ローカル実行: 並べて流す他のジョブを巻き込まないよう pkill はしない (止めるのは timeout)\n")


def rank_out_fix(cfg, s):
    """起動コマンドがランクの標準出力をファイルに書く環境 (site_config.py の rank_out) で、そのファイルを result.txt に足す。
    富士通の mpiexec はランクの標準出力を、起動したシェルのリダイレクト先 (> result.txt) ではなくジョブのログに書くので、
    起動コマンドに -std <rank_out> を付けて書かせる。MPI なし (起動コマンドなし) では何もしない。"""
    f = cfg.get("rank_out")
    if not f or cfg.get("nompi"):
        return s
    import re
    lines = s.split("\n")
    t = [i for i, l in enumerate(lines) if l.lstrip().startswith("timeout -k 30 ")]
    r = [i for i, l in enumerate(lines) if l.strip() == "RC=$?"]
    assert len(t) == 1 and len(r) == 1 and r[0] == t[0] + 1, (t, r)
    ind = re.match(r"\s*", lines[r[0]]).group(0)
    lines[r[0]] += (f"\n{ind}# ランクの標準出力 (起動コマンドが {f} に書く) を result.txt に足す (site_config.py の rank_out)\n"
                    f"{ind}cat {f} >> result.txt 2> /dev/null; rm -f {f}")
    lines[t[0]] = f"{ind}rm -f {f}\n" + lines[t[0]]
    return "\n".join(lines)


def gen(plat, cfg, prec, part, np_):
    """1 ジョブぶんのスクリプトと、そのケース数を返す"""
    s = header(plat, cfg, prec, part, np_)
    tag = prec + (f"_{part[0]}" if part else "")
    # GPU の割り当てスクリプトはジョブごとに書き出すので、np も名前に入れて
    #   同じ精度・分割の np=1/2/4 のジョブが同時に書き換え合わないようにする。
    bindtag = tag + f"_np{np_}"
    build = BUILD_PREBUILT.format(plat=plat, prec=prec, script_dir=SCRIPT_DIR)
    s += BODY.format(plat=plat, prec=prec, nx=NX, version=PARCEL_VERSION,
                     omp=cfg["omp"],
                     launch=launch_for(cfg, np_).replace("TAG", bindtag),
                     build=build,
                     errmax=PRECISIONS[prec]["errmax"],
                     errmax_cases=errmax_cases_for(prec, cfg),
                     gpu_bind=gpu_bind_for(cfg, bindtag))
    s = s.replace("@@PIN_SETUP@@", pin_setup_for(cfg, np_))
    s = local_no_pkill(cfg, s, '    pkill -u $(id -u) -f "$(basename ${BIN})" 2>/dev/null\n', "    ")
    s = rank_out_fix(cfg, s)
    n = 0
    for case, env in cases_for_part(prec, part, cfg):
        s += f'run_case {case} {np_} {" ".join(env)}\n'
        n += 1
    s += '\necho "=== ALL DONE ==="\n'
    return s, n


# ---------------------------------------------------------------- C / Fortran / Python の API の試験 (lang_c / lang_f / lang_py)
#   examples/sample_c.c / sample_f.f90 と tests/regression/sample_py.py をそのまま回帰試験に載せる。何を解いて何と比べるかは
#   試験プログラムのもので、ここでは変えない。C / Fortran は C / Fortran の API から 7 ソルバ (+ pcagmres の implicit 2 通り・
#   ks の shift-invert・旧 API) を呼んで独立に検証し、版 (version) と合わせて 12 項目の PASS / FAIL と最後の
#   "[sample_c] ALL PASS" (Fortran は [sample_f]) を出す。版は PARCEL_EXPECT_VERSION (Include/parcel_version.hpp の値) とも比べる。Python (sample_py.py) は同じ問題・項目・判定を Python のパッケージ parcel から行い、旧 API の項目の
#   代わりに設定を省いた parcel.solve (pcg(defaults)) を見る。先頭にネイティブ層の MPI の有無 (has_mpi) と、GPU の環境
#   (site_config.py の py_device) ではデバイス名 (device) の確認が付く。
#   判定: 1 項目 = 1 ケース。regression_results/<plat>/lang_<c|f|py>/<項目>_np<N>/status.txt に PASS なら OK、FAIL なら FAIL。
#     項目名は記号を _ にしたもの (pcagmres(impl,sym) -> pcagmres_impl_sym)。最後の行と終了コードを 1 ケース (all) と
#     数えるので、1 回の実行で C / Fortran は 13 ケース、Python は 14 ケース (GPU の環境は 15)。出力に無い項目 (途中で落ちた) は
#     TIMEOUT / RUN_FAIL_<rc> / NO_ITEM_OUTPUT。出力全体は regression_results/<plat>/lang_<言語>/_run_np<N>/result.txt
#     (status.txt を置かないので件数に入らない)。
#   並列数: MPI ありは np=1/2/4、MPI なし (<plat>_nompi) は 1 プロセスを直接起動。資源・起動・スレッド数は他の試験と同じ。
#   ビルド: sub_build_lang_<plat>.sh が bin/<plat>/ に sample_c・sample_f・libparcel_py.so を作る (精度ごとのビルドジョブとは別。
#     C / Fortran / Python の都合で verify_* のビルドを止めないため、feed_regression.sh が verify_* の本数でビルドの完了を
#     数えるため)。フラグは verify_* と同じ。対象の環境は site_config.py の REGRESSION_LANG (C / Fortran) と
#     REGRESSION_LANG_PY (Python)。Python は numpy と scipy の入った Python を使う (ビルド時に PARCEL_REG_PY で与え、
#     bin/<plat>/lang_python.txt に記録する。与えなければ python3)。
#   PARCEL をどこから取るか (site_config.py の parcel_from):
#     "source" (既定)  パッケージのソースをその環境の SYCL コンパイラで組む。C / Fortran は capi/ のラッパと組み、
#                      Python は python/src/parcel_py.cpp から bin/<plat>/libparcel_py.so を作ってパッケージの python/ を読む
#                      (PARCEL_TEST_FROM_SOURCE=1、PARCEL_PY_LIB)。
#     "installed"      インストール済みの PARCEL (spack の parcel-sycl / py-parcel-sycl など) を使う。ジョブの環境設定で
#                      PARCEL_PREFIX にその場所を入れ (ENV_SETUP の {parcel_load})、C / Fortran は $PARCEL_PREFIX の
#                      libparcel_sycl_c.so / libparcel_sycl_f.a とヘッダ・モジュールでリンクし、Python は import parcel で読む。
#     検証サンプル (cpp) と ddcheck は、source ではパッケージの Include/ を、installed では $PARCEL_PREFIX/include のヘッダを組む
#     (サンプルが Include/ を相対パスで読むので、installed では試験のソースを写した隣の Include をそこへ向ける。BUILD_INSTALLED_HEADERS)。
LANG_SAMPLE_NX = 16          # sample_c / sample_f / sample_py の既定
LANG_TIMEOUT = 600
# Fortran は項目名を書式 a22 で出すので 22 文字を超える名前は切られる
#   (旧 API の項目 "legacy run_parcel_sycl_f" は "legacy run_parcel_sycl" と表示される。出力の表記に合わせる)。
_CF_ITEMS = ["pcg", "pbicgstab", "pgmres", "pcagmres", "pcagmres(implicit)", "pcagmres(impl,sym)",
             "power", "inverse_power", "ks_arnoldi", "ks_arnoldi(shift-inv)"]
LANGS = {
    "c": dict(title="C の API (examples/sample_c.c)", tag="sample_c",
              items=_CF_ITEMS + ["legacy run_parcel_sycl_c"]),
    "f": dict(title="Fortran の API (examples/sample_f.f90)", tag="sample_f",
              items=_CF_ITEMS + ["legacy run_parcel_sycl"]),
    "py": dict(title="Python (tests/regression/sample_py.py)", tag="sample_py",
               items=_CF_ITEMS + ["pcg(defaults)"]),
}
LANG_ORDER = ["c", "f", "py"]


def parcel_from(cfg):
    v = cfg.get("parcel_from", "source")
    if v not in ("source", "installed"):
        sys.exit(f"parcel_from は source か installed ({v!r})")
    return v


def lang_items(lang, plat):
    """その環境で出力に出るはずの項目名 (出力の順)"""
    items = ["version"] + list(LANGS[lang]["items"])
    if lang == "py":
        cfg = ALL_PLATFORMS[plat]
        items = ["has_mpi"] + (["device"] if cfg.get("py_device") else []) + items
    return items


def lang_key(name):
    """項目名 -> ケース名 (シェルの san () と同じ変換)"""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def langs_for(plat):
    base = ALL_PLATFORMS[plat].get("base", plat)
    out = ["c", "f"] if base in REGRESSION_LANG else []
    if base in REGRESSION_LANG_PY:
        out.append("py")
    return out


LANG_BUILD = """
WORKDIR=${{PBS_O_WORKDIR:-${{PJM_O_WORKDIR:-$(pwd)}}}}
cd ${{WORKDIR}}
if [ ! -f capi/wrapper_parcel_sycl.cpp ]; then
  echo "パッケージのルートで実行すること (capi/wrapper_parcel_sycl.cpp が無い: ${{WORKDIR}})"; exit 1
fi

BINDIR=${{WORKDIR}}/bin/{plat}
LOGDIR=${{WORKDIR}}/buildlog/{plat}/lang
OBJ=${{LOGDIR}}/obj
mkdir -p ${{BINDIR}} ${{OBJ}}
FROM={parcel_from}     # source: パッケージのソースを組む / installed: インストール済みの PARCEL ($PARCEL_PREFIX) を使う
if [ ${{FROM}} = installed ] && [ ! -f "${{PARCEL_PREFIX:-}}/lib/libparcel_sycl_c.so" ]; then
  echo "PARCEL_PREFIX (${{PARCEL_PREFIX:-未設定}}) に lib/libparcel_sycl_c.so が無い (環境設定で spack load などをすること)"; exit 1
fi

# 追加のコンパイルフラグは verify_* のビルド (sub_build_{plat}.sh) と同じものを使う。
#   bin/{plat}/build_extra.txt があればその値を既定にし、PARCEL_BUILD_EXTRA と食い違えば止める。
REC=$(cat ${{BINDIR}}/build_extra.txt 2> /dev/null)
EXTRA=${{PARCEL_BUILD_EXTRA-${{REC}}}}
if [ -f ${{BINDIR}}/build_extra.txt ] && [ "${{REC}}" != "${{EXTRA}}" ]; then
  echo "bin/{plat} の verify_* は PARCEL_BUILD_EXTRA='${{REC}}' で作られている (今回は '${{EXTRA}}')。"; exit 1
fi
[ -n "${{EXTRA}}" ] && echo "  追加フラグ: ${{EXTRA}}"

# 言語は引数で選べる (例: bash {script_dir}/sub_build_lang_{plat}.sh fortran)。c / f (fortran) / py (python)、カンマ区切りでもよい。
#   引数が無ければ PARCEL_LANGS、それも無ければこの環境の全言語 ({langs})。lang_built.txt は選んだ言語の行だけを書き直す。
LANGS=" "
for L in $(echo "${{*:-${{PARCEL_LANGS:-{langs}}}}}" | tr ',' ' '); do
  case ${{L}} in
    c) ;;
    f|fortran) L=f ;;
    py|python) L=py ;;
    *) echo "未知の言語 ${{L}} (c / f (fortran) / py (python))"; exit 1 ;;
  esac
  case " {langs} " in
    *" ${{L}} "*) ;;
    *) echo "{plat} は ${{L}} の試験を持たない (対象: {langs})"; exit 1 ;;
  esac
  case "${{LANGS}}" in *" ${{L}} "*) ;; *) LANGS="${{LANGS}}${{L}} " ;; esac
done
CC="{cc}"
FC="{fc}"
PY=${{PARCEL_REG_PY:-python3}}
# 作れた言語の一覧 (lang_built.txt) から今回の言語を外す (作れたら最後に足す。他の言語の記録は残す)
if [ -f ${{BINDIR}}/lang_built.txt ]; then
  for L in ${{LANGS}}; do
    grep -vx "${{L}}" ${{BINDIR}}/lang_built.txt > ${{BINDIR}}/lang_built.tmp
    mv ${{BINDIR}}/lang_built.tmp ${{BINDIR}}/lang_built.txt
  done
fi

step () {{   # $1=名前 $2...=コマンド (ログは buildlog/{plat}/lang/<名前>.log)
  local NAME=$1; shift
  "$@" > ${{LOGDIR}}/${{NAME}}.log 2>&1
  local RC=$?
  if [ ${{RC}} -ne 0 ]; then
    echo "  [FAIL] ${{NAME}} (rc=${{RC}}, ${{LOGDIR}}/${{NAME}}.log)"
    grep -E "error|Error" ${{LOGDIR}}/${{NAME}}.log | head -3 | sed 's/^/         /'
  fi
  return ${{RC}}
}}

# Fortran の実行時ライブラリ (MPI ありは MPI の Fortran ライブラリも)。SYCL コンパイラでリンクするので明示する
fc_link () {{
  local L=""
  if [ "{mpi}" = 1 ]; then
    if ${{FC}} --showme:link > /dev/null 2>&1; then L=$(${{FC}} --showme:link)                 # Open MPI
    else L=$(${{FC}} -show | tr ' ' '\\n' | grep -E '^-[Ll]' | tr '\\n' ' ')                    # Intel MPI など
    fi
  fi
  echo "${{L}} {fc_rt}"
}}

echo "=== {plat} の C / Fortran / Python の試験プログラムを作る (言語:${{LANGS}}、PARCEL: ${{FROM}}) ==="
T0=$(date +%s)
NC=0; NF=0; NP_=0
case "${{LANGS}}" in *" c "*) if [ -x ${{BINDIR}}/sample_c ]; then echo "  [skip] sample_c (既にある)"; else NC=1; fi ;; esac
case "${{LANGS}}" in *" f "*) if [ -x ${{BINDIR}}/sample_f ]; then echo "  [skip] sample_f (既にある)"; else NF=1; fi ;; esac
case "${{LANGS}}" in *" py "*) NP_=1 ;; esac
if [ ${{FROM}} = source ]; then
  # ラッパ (C++) を SYCL コンパイラで、main を C / Fortran のコンパイラで作り、SYCL コンパイラでリンクする
  #   (examples/build_examples.sh と同じ形)。SYCL コンパイラ (clang) は PIE でリンクするので main は -fPIE で作る。
  if [ ${{NC}} = 1 ] || [ ${{NF}} = 1 ]; then
    # 旧 API run_parcel_sycl_c/_f は反復上限を MAX_ITERATION で決める (既定 10000。明示しておく)
    step wrapper ${{CXX}} -O3 {target} -DMAX_ITERATION=10000 ${{EXTRA}} {lapack} \\
        -c capi/wrapper_parcel_sycl.cpp -o ${{OBJ}}/wrapper_parcel_sycl.o || exit 1
  fi
  if [ ${{NC}} = 1 ]; then
    step sample_c.o ${{CC}} -O2 -fPIE {mpidef}-Icapi -c examples/sample_c.c -o ${{OBJ}}/sample_c.o &&
    step sample_c ${{CXX}} {target} -o ${{BINDIR}}/sample_c ${{OBJ}}/sample_c.o ${{OBJ}}/wrapper_parcel_sycl.o \\
        {lapack} -lm ${{LINKOPT}}
    [ -x ${{BINDIR}}/sample_c ] && echo "  [OK]   sample_c  $(( $(date +%s)-T0 )) 秒"
  fi
  if [ ${{NF}} = 1 ]; then
{fstub}    step parcel_sycl_mod.o ${{FC}} -O2 -fPIE {fmod}${{OBJ}} -c capi/parcel_sycl_mod.f90 -o ${{OBJ}}/parcel_sycl_mod.o &&
    step sample_f.o ${{FC}} -O2 -fPIE -I${{OBJ}} {fmod}${{OBJ}} -c examples/sample_f.f90 -o ${{OBJ}}/sample_f.o &&
    step sample_f ${{CXX}} {target} -o ${{BINDIR}}/sample_f ${{OBJ}}/sample_f.o {fstubobj}${{OBJ}}/parcel_sycl_mod.o \\
        ${{OBJ}}/wrapper_parcel_sycl.o $(fc_link) {lapack} ${{LINKOPT}}
    [ -x ${{BINDIR}}/sample_f ] && echo "  [OK]   sample_f  $(( $(date +%s)-T0 )) 秒"
  fi
  # Python のネイティブ層 (python/CMakeLists.txt と同じ形を pip を通さずに)。フラグは verify_* と同じ
  if [ ${{NP_}} = 1 ]; then
    if [ -f ${{BINDIR}}/libparcel_py.so ]; then echo "  [skip] libparcel_py.so (既にある)"
    else
      step libparcel_py ${{CXX}} -O3 {target} ${{EXTRA}} -fPIC -shared -IInclude {lapack} \\
          -o ${{BINDIR}}/libparcel_py.so python/src/parcel_py.cpp ${{LINKOPT}} &&
      [ -f ${{BINDIR}}/libparcel_py.so ] && echo "  [OK]   libparcel_py.so  $(( $(date +%s)-T0 )) 秒"
    fi
  fi
else
  # インストール済みの PARCEL: C / Fortran の main をそのライブラリとリンクする (spack/stack/check_stack.sh と同じ形)
  P=${{PARCEL_PREFIX}}
  echo "  PARCEL_PREFIX=${{P}}"
  if [ ${{NC}} = 1 ]; then
    step sample_c ${{CC}} -O2 {mpidef}-I${{P}}/include -o ${{BINDIR}}/sample_c examples/sample_c.c \\
        -L${{P}}/lib -lparcel_sycl_c -Wl,-rpath,${{P}}/lib -lm
    [ -x ${{BINDIR}}/sample_c ] && echo "  [OK]   sample_c  $(( $(date +%s)-T0 )) 秒"
  fi
  if [ ${{NF}} = 1 ]; then
    # モジュール (.mod) は ${{OBJ}} に書く (-J)。カレント (パッケージのルート) に書くと、gfortran は -I・-J より先に
    #   カレントを探すので、MPI なしの構成が作った代用の mpi.mod を後の MPI ありの構成が読んでリンクに失敗する
    step sample_f ${{FC}} -O2 -I${{P}}/include {fmod}${{OBJ}} -o ${{BINDIR}}/sample_f {fstubsrc}examples/sample_f.f90 \\
        -L${{P}}/lib -lparcel_sycl_f -lparcel_sycl_c -Wl,-rpath,${{P}}/lib
    [ -x ${{BINDIR}}/sample_f ] && echo "  [OK]   sample_f  $(( $(date +%s)-T0 )) 秒"
  fi
fi
# 試験に使う Python (numpy と scipy、installed では parcel も読めること)。ジョブはここに記録した Python を使う
#   (qsub -v PARCEL_REG_PY=... などで上書きできる)
if [ ${{NP_}} = 1 ]; then
  CHK="import numpy, scipy"; [ ${{FROM}} = installed ] && CHK="${{CHK}}, parcel; print(parcel.build_info())"
  if {py_launch}${{PY}} -c "${{CHK}}" > ${{LOGDIR}}/python_check.log 2>&1; then
    command -v ${{PY}} > ${{BINDIR}}/lang_python.txt
    echo "  [OK]   python $(cat ${{BINDIR}}/lang_python.txt) ($(tail -1 ${{LOGDIR}}/python_check.log))"
  else
    echo "  [FAIL] python ${{PY}}: ${{CHK}} が通らない (${{LOGDIR}}/python_check.log)。PARCEL_REG_PY で与えること"
    rm -f ${{BINDIR}}/lang_python.txt
  fi
fi
# 成果物で確かめる (コマンドの終了コードだけで判断しない)。feed_regression.sh の lang はこの一覧を待つ
for L in ${{LANGS}}; do
  case ${{L}} in
    c|f) F=${{BINDIR}}/sample_${{L}} ;;
    py)  F=${{BINDIR}}/lang_python.txt; [ ${{FROM}} = source ] && [ ! -f ${{BINDIR}}/libparcel_py.so ] && F=/nonexistent ;;
  esac
  if [ -e ${{F}} ]; then echo ${{L}} >> ${{BINDIR}}/lang_built.txt; else echo "  MISSING ${{L}} (${{F}})"; fi
done
echo "  作れた言語: $(tr '\\n' ' ' < ${{BINDIR}}/lang_built.txt 2> /dev/null)"
echo "=== ALL DONE ==="
"""


def lang_build_job(plat, cfg):
    """1 環境ぶんの C / Fortran / Python のビルドジョブ"""
    langs = langs_for(plat)
    s = header(plat, cfg, "build", None, NPROC_LIST[-1], build=True)
    s = s.replace("PARCEL_R_" + plat + "_build_np" + str(NPROC_LIST[-1]), "PARCEL_L_" + plat)
    s = s.replace(f"regression_{plat}_build_np{NPROC_LIST[-1]}", f"build_lang_{plat}")
    old = (f"# 回帰試験: {plat} / 精度 build\n"
           f"#   NX={NX}, MPI = {nps_text(cfg, NPROC_LIST[-1])}\n"
           f"#   ソルバ・行列・前処理は実行時オプションで切り替える"
           f"のでビルド不要 (bin/ の事前ビルドを使用)\n")
    assert s.count(old) == 1, plat
    where = ("計算ノードで実行する (pjsub で投入)" if cfg["sched"] == "tcs" and is_ody(cfg)
             else "ログインノードで bash で実行してよい (qsub でも動く)")
    made = {"c": "sample_c", "f": "sample_f", "py": "libparcel_py.so"}
    s = s.replace(old,
                  f"# C / Fortran / Python の API の試験プログラムを作る: {plat}\n"
                  f"#   bin/{plat}/ に {', '.join(made[l] for l in langs)} を作る (フラグは verify_* と同じ)。"
                  f"ログは buildlog/{plat}/lang/。\n"
                  f"#   試験ジョブ (sub_regression_{plat}_lang_*.sh) より先に済ませること。{where}。\n"
                  f"#   出来ているものは飛ばす。作り直すときは bin/{plat}/ の該当ファイルを消してから流す。\n"
                  + ("#   Python は numpy と scipy の入った Python を PARCEL_REG_PY で与える (既定 python3)。\n"
                     if "py" in langs else ""))
    nompi = bool(cfg.get("nompi"))
    base = cfg.get("base", plat)
    fmod = cfg.get("lang_fmod", "-J")      # Fortran のモジュールの出力先のオプション (富士通の frt は -M)
    pick = lambda v: v.get(base, v["default"]) if isinstance(v, dict) else v
    if nompi:
        cc, fc, mpidef = pick(LANG_NOMPI_CC), pick(LANG_NOMPI_FC), "-DPARCEL_MPI=0 "
        # MPI なしの Fortran は 1 プロセス用の mpi モジュール (examples/sample_f_mpi_stub.f90) と組む
        fstub = (f"    step sample_f_mpi_stub.o ${{FC}} -O2 -fPIE {fmod}${{OBJ}} -c examples/sample_f_mpi_stub.f90 "
                 "-o ${OBJ}/sample_f_mpi_stub.o &&\n")
        fstubobj = "${OBJ}/sample_f_mpi_stub.o "
        fstubsrc = "examples/sample_f_mpi_stub.f90 "
    else:
        cc, fc = pick(LANG_MPI_CC), pick(LANG_MPI_FC)
        mpidef = fstub = fstubobj = fstubsrc = ""
    s += LANG_BUILD.format(plat=plat, cc=cc, fc=fc, mpi=0 if nompi else 1, mpidef=mpidef, langs=" ".join(langs),
                           fstub=fstub, fstubobj=fstubobj, fstubsrc=fstubsrc, fmod=fmod, target=target_of(cfg),
                           # MPI ありの parcel を読む確認を起動コマンド経由で行う (富士通 MPI は mpiexec なしで MPI を初期化できない)
                           py_launch="" if nompi else cfg.get("py_check_launch", ""),
                           lapack=lapack_of(cfg), script_dir=SCRIPT_DIR, parcel_from=parcel_from(cfg),
                           # gfortran の実行時ライブラリ (site_config.py の lang_fc_rt で置き換えられる。macOS は .dylib)
                           fc_rt=cfg.get("lang_fc_rt",
                                         "-L$(dirname $(${FC} -print-file-name=libgfortran.so)) -lgfortran"))
    return s


LANG_BODY = """
cd ${{WORKDIR}}
OUT=regression_results/{plat}/lang_{lang}
RUN=${{OUT}}/_run_np{np}          # 出力全体 (status.txt は置かない)
NP={np}      # 起動コマンドが参照する
SUMD=${{OUT}}/all_np{np}
# 前回 OK で終わっていれば飛ばす (PARCEL_REG_FORCE=1 でやり直す)
if [ -z "${{PARCEL_REG_FORCE:-}}" ] && [ "$(cat ${{SUMD}}/status.txt 2>/dev/null)" = "OK" ]; then
  echo "  [skip] lang_{lang} np={np}"
  echo "=== ALL DONE ==="
  exit 0
fi
# この並列数の前回の項目を消す (項目が減ったときに古い結果が残らないように)
rm -rf ${{OUT}}/*_np{np} ${{RUN}}
mkdir -p ${{RUN}} ${{SUMD}}
# ケース名: 記号を _ にする (gen_regression.py の lang_key と同じ)
san () {{ echo "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^_+//; s/_+$//'; }}
EXPECT="{expect}"
missing () {{   # $1=無かったもの
  echo "  BUILD_MISSING $1"; echo "  先に {script_dir}/sub_build_lang_{plat}.sh を実行すること"
  for K in ${{EXPECT}} all; do mkdir -p ${{OUT}}/${{K}}_np{np}; echo BUILD_MISSING > ${{OUT}}/${{K}}_np{np}/status.txt; done
  exit 1
}}
{check}
echo "=== {title} {plat} np={np} ==="
{gpu_bind}
@@PIN_SETUP@@
export OMP_NUM_THREADS={omp}
export PARALLEL=${{OMP_NUM_THREADS}}
{lang_env}cd ${{RUN}}
T_START=$(date '+%F %T')
timeout -k 30 {tmo} {launch} {cmd} > result.txt 2>&1 < /dev/null
RC=$?
echo "host=$(hostname) job=${{PBS_JOBID:-${{PJM_JOBID:-}}}} start=${{T_START}} end=$(date '+%F %T') rc=${{RC}} tmo={tmo} acpp_warnings=$(grep -c 'AdaptiveCpp Warning' result.txt)" > run_info.txt
cd ${{WORKDIR}}
RES=${{RUN}}/result.txt
[ ${{RC}} -eq 124 ] && pkill -u $(id -u) -f "{pkill}" 2>/dev/null

# 項目ごとのケース: "状態<TAB>項目名<TAB>行番号" に直して 1 項目 1 ディレクトリにする
#   (置き換えは awk で行う。BSD sed (macOS) は置き換え側の \\t をタブにしない)
grep -nE '^\\[{tag}\\] +.*[^ ] +(PASS|FAIL)  rc=' ${{RES}} | \\
    awk '{{n = $0; sub(/:.*/, "", n); l = substr($0, length(n) + 2); sub(/^\\[{tag}\\] +/, "", l);
          if (match(l, / +(PASS|FAIL)  rc=/)) {{st = substr(l, RSTART, RLENGTH); gsub(/ |rc=/, "", st);
          printf "%s\\t%s\\t%s\\n", st, substr(l, 1, RSTART - 1), n}}}}' | while IFS=$'\\t' read -r ST NAME LN; do
  K=$(san "${{NAME}}"); K0=${{K}}; i=2
  while [ -d ${{OUT}}/${{K}}_np{np} ]; do K=${{K0}}_${{i}}; i=$((i+1)); done   # 同じ名前は _2, _3 ...
  D=${{OUT}}/${{K}}_np{np}; mkdir -p ${{D}}
  sed -n "${{LN}}p" ${{RES}} > ${{D}}/result.txt
  sed -n "$((LN+1))p" ${{RES}} | grep -E '^ +converged_reason' >> ${{D}}/result.txt
  cp ${{RUN}}/run_info.txt ${{D}}/
  if [ "${{ST}}" = PASS ]; then echo OK > ${{D}}/status.txt; else echo FAIL > ${{D}}/status.txt; fi
done
# 出力に無い項目 (途中で落ちた・打ち切った) も NG として数える
if [ ${{RC}} -eq 124 ]; then MISS=TIMEOUT
elif [ ${{RC}} -ne 0 ]; then MISS=RUN_FAIL_${{RC}}
else MISS=NO_ITEM_OUTPUT
fi
for K in ${{EXPECT}}; do
  D=${{OUT}}/${{K}}_np{np}
  [ -f ${{D}}/status.txt ] && continue
  mkdir -p ${{D}}; echo ${{MISS}} > ${{D}}/status.txt; cp ${{RUN}}/run_info.txt ${{D}}/
done
# 最後の行 (ALL PASS / FAILED) と終了コード
grep -E '^\\[{tag}\\] (ALL PASS|FAILED)' ${{RES}} | tail -1 > ${{SUMD}}/result.txt
cp ${{RUN}}/run_info.txt ${{SUMD}}/
if [ ${{RC}} -eq 124 ]; then echo TIMEOUT > ${{SUMD}}/status.txt
elif [ ${{RC}} -eq 0 ] && grep -q '^\\[{tag}\\] ALL PASS' ${{RES}}; then echo OK > ${{SUMD}}/status.txt
elif grep -q '^\\[{tag}\\] FAILED (' ${{RES}}; then echo FAIL > ${{SUMD}}/status.txt
elif [ ${{RC}} -ne 0 ]; then echo RUN_FAIL_${{RC}} > ${{SUMD}}/status.txt
else echo NO_SUMMARY > ${{SUMD}}/status.txt
fi
for f in ${{OUT}}/*_np{np}/status.txt; do
  echo "  [$(cat $f)] $(basename $(dirname $f))"
done
echo "  lang_{lang} np={np}: $(cat ${{OUT}}/*_np{np}/status.txt | sort | uniq -c | awk '{{printf "%s=%s ", $2, $1}}')  (rc=${{RC}})"

echo "=== ALL DONE ==="
"""


def gen_lang(plat, cfg, lang, np_):
    """C / Fortran / Python の API の試験 (1 環境・1 言語・1 並列数) のジョブを返す"""
    L = LANGS[lang]
    s = header(plat, cfg, f"lang_{lang}", None, np_)
    old = (f"# 回帰試験: {plat} / 精度 lang_{lang}\n"
           f"#   NX={NX}, MPI = {nps_text(cfg, np_)}\n"
           f"#   ソルバ・行列・前処理は実行時オプションで切り替える"
           f"のでビルド不要 (bin/ の事前ビルドを使用)\n")
    new = (f"# 回帰試験 {L['title']}: {plat} / MPI = {nps_text(cfg, np_)}\n"
           f"#   1 項目 = 1 ケース (regression_results/{plat}/lang_{lang}/<項目>_np{np_}/status.txt)。\n"
           f"#   試験プログラムは sub_build_lang_{plat}.sh が bin/{plat}/ に作る。\n")
    assert s.count(old) == 1, (plat, lang)
    s = s.replace(old, new)
    bindtag = f"lang_{lang}_np{np_}"
    if lang in ("c", "f"):
        check = (f"BIN=${{WORKDIR}}/bin/{plat}/{L['tag']}\n"
                 'if [ ! -x "${BIN}" ]; then missing ${BIN}; fi\nls -l ${BIN}')
        cmd = f"${{BIN}} {LANG_SAMPLE_NX}"
        lang_env = f"export PARCEL_EXPECT_VERSION={PARCEL_VERSION}\n"
        pkill = L["tag"]
    else:
        src = parcel_from(cfg) == "source"
        check = (f"PY=${{PARCEL_REG_PY:-$(cat ${{WORKDIR}}/bin/{plat}/lang_python.txt 2> /dev/null)}}\n"
                 '[ -n "${PY}" ] || missing "bin/' + plat + '/lang_python.txt"\n'
                 + (f"LIB=${{WORKDIR}}/bin/{plat}/libparcel_py.so\n"
                    '[ -f "${LIB}" ] || missing ${LIB}\nls -l ${LIB}\n' if src else "")
                 + 'echo "  python: ${PY}"')
        cmd = f"${{PY}} ${{WORKDIR}}/{SCRIPT_DIR}/sample_py.py {LANG_SAMPLE_NX}"
        dev = cfg.get("py_device")
        lang_env = (("# パッケージの python/ と bin/ のネイティブ層を使う\n"
                     "export PARCEL_TEST_FROM_SOURCE=1\nexport PARCEL_PY_LIB=${LIB}\n") if src else
                    "# インストール済みの parcel を使う (環境設定で読み込んだもの)\nunset PARCEL_TEST_FROM_SOURCE PARCEL_PY_LIB\n")
        lang_env += (f"export PARCEL_EXPECT_MPI={0 if cfg.get('nompi') else 1}\n"
                     + (f"export PARCEL_EXPECT_DEVICE='{dev}'\n" if dev else "")
                     + f"export PARCEL_EXPECT_VERSION={PARCEL_VERSION}\n")
        pkill = f"{SCRIPT_DIR}/sample_py.py"
    s += LANG_BODY.format(
        plat=plat, lang=lang, np=np_, title=L["title"], tag=L["tag"], script_dir=SCRIPT_DIR,
        expect=" ".join(lang_key(i) for i in lang_items(lang, plat)), omp=cfg["omp"], tmo=LANG_TIMEOUT,
        check=check, cmd=cmd, lang_env=lang_env, pkill=pkill,
        launch=launch_for(cfg, np_).replace("TAG", bindtag),
        gpu_bind=gpu_bind_for(cfg, bindtag))
    s = s.replace("@@PIN_SETUP@@", pin_setup_for(cfg, np_))
    # 起動コマンドが空 (MPI なし) のときの余分な空白をなくす
    s = s.replace(f"timeout -k 30 {LANG_TIMEOUT}  ", f"timeout -k 30 {LANG_TIMEOUT} ")
    s = local_no_pkill(cfg, s, f'[ ${{RC}} -eq 124 ] && pkill -u $(id -u) -f "{pkill}" 2>/dev/null\n', "")
    return rank_out_fix(cfg, s)


def write_script(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


# ---------------------------------------------------------------- 選んで生成する
#   MPI あり / なしと言語 (cpp / c / fortran / python) を任意に組み合わせて生成できる。
#     python3 tests/regression/gen_regression.py                                全環境・MPI あり/なし・全言語
#     python3 tests/regression/gen_regression.py s86_cpu s86_cpu_nompi          環境を並べる (従来の書き方)
#     python3 tests/regression/gen_regression.py --plat s86_cpu --mpi no --lang c   s86_cpu_nompi の C だけ
#   選んだときは選んだ組の生成物だけを消して作り直す (他の環境・言語・MPI の有無のスクリプトは残す)。
#   何も選ばないときは sub_regression_*.sh / sub_build_*.sh をすべて消してから作る。
LANG_ALIASES = {"cpp": "cpp", "c++": "cpp", "cxx": "cpp", "c": "c", "f": "f", "fortran": "f",
                "py": "py", "python": "py"}
ALL_LANG_KEYS = ["cpp", "c", "f", "py"]


def parse_langs(text):
    out = []
    for t in text.replace(" ", ",").split(","):
        if not t:
            continue
        if t.lower() not in LANG_ALIASES:
            sys.exit(f"未知の言語 {t} (cpp / c / fortran / python)")
        k = LANG_ALIASES[t.lower()]
        if k not in out:
            out.append(k)
    return out


def select_plats(names, mpi):
    """環境名の並び (MPI ありの名前でも <plat>_nompi でもよい) と --mpi から、生成する環境を ALL_PLATFORMS の順で返す"""
    bases = []
    for t in names:
        b = t[:-len(NOMPI_SUFFIX)] if t.endswith(NOMPI_SUFFIX) else t
        if b not in ALL_PLATFORMS and b + NOMPI_SUFFIX not in ALL_PLATFORMS:
            sys.exit(f"未知の環境 {t} (tests/site_config.py の REGRESSION と REGRESSION_NOMPI から作るもの: "
                     f"{' '.join(ALL_PLATFORMS)})")
        if b not in bases:
            bases.append(b)
    out = []
    for plat, cfg in ALL_PLATFORMS.items():
        if bases and cfg.get("base", plat) not in bases:
            continue
        is_nompi = bool(cfg.get("nompi"))
        if (mpi == "yes" and is_nompi) or (mpi == "no" and not is_nompi):
            continue
        out.append(plat)
    return out


def remove_selected(plats, langs):
    """選んだ組 (環境 x 言語) の生成物だけを消す (分割が減ったときに古いスクリプトが残って一緒に流れないように)"""
    n = 0
    for plat in plats:
        paths = []
        if "cpp" in langs:
            paths.append(os.path.join(HERE, f"sub_build_{plat}.sh"))
            pre = f"sub_regression_{plat}_"
            for f in glob.glob(os.path.join(HERE, pre + "*.sh")):
                # 同じ名前で始まる別の環境 (<plat>_nompi) と言語別 (_lang_) は除く
                if not os.path.basename(f)[len(pre):].startswith(("nompi_", "lang_")):
                    paths.append(f)
        for l in langs:
            if l != "cpp":
                paths += glob.glob(os.path.join(HERE, f"sub_regression_{plat}_lang_{l}_np*.sh"))
        for f in paths:
            if os.path.exists(f):
                os.remove(f); n += 1
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="回帰試験のジョブスクリプトを生成する (何も選ばなければ全部)")
    ap.add_argument("plats", nargs="*", help="環境 (--plat と同じ。従来の書き方)")
    ap.add_argument("--plat", default="", help="環境 (カンマ区切り。MPI ありの名前を与えれば --mpi で MPI なしも選べる)")
    ap.add_argument("--mpi", choices=("yes", "no", "both"), default=None,
                    help="yes = MPI あり (<plat>)、no = MPI なし (<plat>_nompi)、both = 両方。"
                         "既定は、環境を並べたときはその名前そのまま、並べなければ両方")
    ap.add_argument("--lang", default="cpp,c,fortran,python", help="言語 (カンマ区切り: cpp / c / fortran / python)。既定は全部")
    args = ap.parse_args(argv)
    names = list(args.plats) + [t for t in args.plat.replace(" ", ",").split(",") if t]
    langs = parse_langs(args.lang)
    for p in names:
        if p not in ALL_PLATFORMS and p + NOMPI_SUFFIX not in ALL_PLATFORMS:
            select_plats([p], "both")            # 未知の環境の案内を出して止める
    if names and args.mpi is None:
        plats = [p for p in ALL_PLATFORMS if p in names]        # 並べた名前そのまま (従来の動作)
    else:
        plats = select_plats(names, args.mpi or "both")
    if not names and args.mpi in (None, "both") and set(langs) == set(ALL_LANG_KEYS):
        # 前回の生成物を消す (分割数が減ったときに古いスクリプトが残って一緒に投入されないように)
        for pat in ("sub_regression_*.sh", "sub_build_*.sh"):
            for f in sorted(glob.glob(os.path.join(HERE, pat))):
                os.remove(f)
    else:
        nrm = remove_selected(plats, langs)
        print(f"  選択: 環境 {' '.join(plats)} / 言語 {','.join(langs)} (選んだ組の古い生成物 {nrm} 本を消した)")
    total_jobs = total_cases = 0
    per_plat = {}
    if "cpp" in langs:
        for plat in plats:
            cfg = ALL_PLATFORMS[plat]
            njob = ncase = 0
            write_script(os.path.join(HERE, f"sub_build_{plat}.sh"), build_job(plat, cfg))
            for prec in PRECISIONS:
                for part in SPLIT + [PSPEC_PART, CHOLQR_PART, KSQR_PART]:
                    for np_ in nprocs_of(cfg):
                        s, n = gen(plat, cfg, prec, part, np_)
                        if n == 0:
                            continue
                        write_script(os.path.join(
                            HERE, f"sub_regression_{plat}_{prec}_{part[0]}_np{np_}.sh"), s)
                        total_jobs += 1; njob += 1
                        total_cases += n; ncase += n
            for np_ in [q for q in DDCHECK_NP if q in nprocs_of(cfg)]:
                write_script(os.path.join(HERE, f"sub_regression_{plat}_ddcheck_np{np_}.sh"),
                             gen_ddcheck(plat, cfg, np_))
                total_jobs += 1; njob += 1
                total_cases += 1; ncase += 1
            per_plat[plat] = (njob, ncase)
        print(f"  {'環境':<16} ジョブ数  ケース数")
        for plat, (njob, ncase) in per_plat.items():
            print(f"  {plat:<16} {njob:>6}  {ncase:>7}")
        print(f"  {'合計':<16} {total_jobs:>6} ジョブ / {total_cases} ケース"
              f"  + ビルド {len(plats)} 本")
    # C / Fortran / Python の API の試験 (REGRESSION_LANG / REGRESSION_LANG_PY の環境だけ)。ビルド (sub_build_lang_<plat>.sh) は
    #   言語を選んでも中身が同じ (その環境の全言語を既定に持ち、引数で絞る) なので、言語を 1 つでも選べば書く
    lsel = [l for l in langs if l != "cpp"]
    nlang = nbl = 0
    nlang_of = {}
    for plat in plats:
        cfg = ALL_PLATFORMS[plat]
        ls = [l for l in langs_for(plat) if l in lsel]
        if not ls:
            continue
        write_script(os.path.join(HERE, f"sub_build_lang_{plat}.sh"), lang_build_job(plat, cfg))
        nbl += 1
        for lang in ls:
            for np_ in nprocs_of(cfg):
                write_script(os.path.join(HERE, f"sub_regression_{plat}_lang_{lang}_np{np_}.sh"),
                             gen_lang(plat, cfg, lang, np_))
                nlang += 1
                nlang_of[lang] = nlang_of.get(lang, 0) + 1
    if nbl:
        per = "、".join(f"{ {'c': 'C', 'f': 'Fortran', 'py': 'Python'}[l] } {nlang_of[l]} 本"
                       for l in LANG_ORDER if l in nlang_of)
        print(f"  C / Fortran / Python の API: {nbl} 環境、ジョブ {nlang} 本 ({per}) + ビルド {nbl} 本"
              f" (sub_build_lang_<環境>.sh)")


if __name__ == "__main__":
    main()
