# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""PARCEL (SYCL 版) の Python パッケージ parcel の spack レシピ。"""
import os

from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.python import PythonPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage
from spack_repo.parcel.packages.parcel_sycl.package import (acpp_settings, fj_env, hwloc_env, lapack_ldflags,
                                                      lapacke_include, mpicxx, quick_test_launcher,
                                                      quick_test_run, quick_test_skip_reason, rpath_flags)

from spack.package import *


class PyParcelSycl(PythonPackage, CudaPackage, ROCmPackage):
    """PARCEL (SYCL 版) の Python インタフェース (import parcel)。ネイティブ層 (libparcel_py.so) を
    配布パッケージの python/ (scikit-build-core) からビルドする。既定は MPI なしの 1 プロセス版。"""

    homepage = "https://ccse2.jaea.go.jp/software/parcel/"
    # 配布 tarball の URL (GitHub の Release。tools/make_tarball.py が作る parcel_sycl-<版>.tar.gz)。
    #   Release に置く前は `spack develop py-parcel-sycl@1.2.0 path=<配布パッケージ>` で使うか、tarball を手元の mirror に置いて使う
    #   (spack/README.md の「配布 tarball から入れる」)。
    url = "https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz"
    # 版と tarball の sha256。sha256 は tools/make_tarball.py --pin が書く (tarball に入れるレシピではこの行を
    #   sha256 なしの形に戻すので、書いた後も tarball と sha256 は変わらない。tools/make_tarball.py --check で照合)。
    #   版は Include/parcel_version.hpp の MAJOR.MINOR.PATCH と同じにする (tests/check_version.py で照合)
    version("1.2.0", sha256="470acf98f3b634a15fb0e866ad8633e5a3474f7c5764aa82028669fe6119a2ce")
    build_directory = "python"

    license("LGPL-3.0-only")

    variant("mpi", default=False, description="MPI あり (既定は MPI なしの 1 プロセス版)")
    variant("native", default=True, description="ホスト向けに -march=native -mtune=native を付ける (+fj では付けない)")
    variant("fj", default=False, description="A64FX: ホスト側を富士通コンパイラ (FCC) でコンパイルする")
    variant("kernel", default="auto", values=("auto", "blocking_dot", "a64fx", "ladder", "none"),
            description="カーネル最適化 (auto: CPU は blocking_dot, NVIDIA の GPU (+cuda) は none, AMD の GPU (+rocm) は ladder)")
    variant("ks_ortho", default="cgs2", values=("cgs2", "mgs"),
            description="ks_arnoldi の直交化の既定 (set_qr_algo() を呼ばないとき)。mgs は -DPARCEL_KS_ORTHO_DEFAULT=mgs (PS-A で使う)")
    variant("scipy", default=True,
            description="SciPy も入れる (DistributedCSR.from_scipy / from_global と examples/quick_start.py が使う。pip の parcel[scipy] に当たる)")
    conflicts("+cuda", when="+rocm", msg="+cuda と +rocm は同時に指定できない")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-scikit-build-core+pyproject@0.10:", type="build")
    depends_on("cmake@3.18:", type="build")
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy", type="run", when="+scipy")
    depends_on("hipsycl", type=("build", "link", "run"))
    # AdaptiveCpp の LLVM のライブラリ (libomp.so など) は新しい形式の圧縮デバッグ情報を持ち, OS 付属の古い ld では読めない
    depends_on("binutils", type=("build", "run"), when="platform=linux")   # macOS の ld は Apple のもの (GNU の ld は Mach-O を扱えない)
    depends_on("hipsycl+cuda", when="+cuda")
    depends_on("hipsycl+rocm", when="+rocm")
    depends_on("lapack", when="~fj", type=("build", "link", "run"))
    depends_on("mpi", when="+mpi ~fj", type=("build", "link", "run"))   # +fj は富士通 MPI (mpiFCC) を直接使う
    # ヘッダ (Include/) は parcel-sycl から取る。MPI の有無とターゲットは同じにする
    depends_on("parcel-sycl", type="build")
    depends_on("parcel-sycl~mpi", when="~mpi", type="build")
    depends_on("parcel-sycl+mpi", when="+mpi", type="build")
    for _a in CudaPackage.cuda_arch_values:
        depends_on("parcel-sycl+cuda cuda_arch=%s" % _a, when="+cuda cuda_arch=%s" % _a, type="build")
    for _a in ROCmPackage.amdgpu_targets:
        depends_on("parcel-sycl+rocm amdgpu_target=%s" % _a, when="+rocm amdgpu_target=%s" % _a, type="build")
    depends_on("parcel-sycl~cuda~rocm", when="~cuda~rocm", type="build")
    depends_on("fj", when="+fj", type=("build", "run"))
    depends_on("parcel-sycl+fj", when="+fj", type="build")
    depends_on("parcel-sycl~fj", when="~fj", type="build")
    depends_on("parcel-sycl ks_ortho=cgs2", when="ks_ortho=cgs2", type="build")
    depends_on("parcel-sycl ks_ortho=mgs", when="ks_ortho=mgs", type="build")

    def _acpp(self):
        return join_path(self.spec["hipsycl"].prefix.bin, "acpp")

    def setup_build_environment(self, env):
        self._env(env)

    def setup_run_environment(self, env):
        self._env(env)
        hwloc_env(self.spec, env)

    def _env(self, env):
        # parcel-sycl がビルド時の ACPP_CUDA_PATH を記録していれば同じものを使う
        rec = join_path(self.spec["parcel-sycl"].prefix, "share", "parcel", "acpp_cuda_path") \
            if "parcel-sycl" in self.spec else None
        if rec and os.path.isfile(rec):
            env.set("ACPP_CUDA_PATH", open(rec).read().strip())
        if self.spec.satisfies("+mpi"):
            for v in ("OMPI_CXX", "MPICH_CXX", "I_MPI_CXX"):
                env.set(v, self._acpp())
        fj_env(self.spec, env)

    def _lapacke_dir(self, spec):
        # parcel-sycl が写した LAPACKE のヘッダを優先する
        d = join_path(spec["parcel-sycl"].prefix.include, "lapacke")
        return d if os.path.isdir(d) else lapacke_include(spec)

    def config_settings(self, spec, prefix):
        target, kflags, hflags, mflags = acpp_settings(self, spec)
        return {
            # ビルド用のディレクトリは spack の作業ディレクトリに置く (既定の python/build/ はソースの中にあり,
            #   同じソースを使う別の構成・別の環境のビルドと共有されて, 他の構成のライブラリが入ったことがある)
            "build-dir": join_path(self.stage.path, "skbuild"),
            "cmake.define.PARCEL_MPI": "1" if spec.satisfies("+mpi") else "0",
            "cmake.define.PARCEL_MPICXX": mpicxx(spec) if spec.satisfies("+mpi") else self._acpp(),
            "cmake.define.PARCEL_TARGET": target,
            "cmake.define.PARCEL_KERNEL_FLAGS": " ".join(kflags),
            "cmake.define.PARCEL_HOST_FLAGS": " ".join(hflags),
            "cmake.define.PARCEL_INCLUDE_DIR": spec["parcel-sycl"].prefix.include,
            "cmake.define.LAPACKE_INCLUDE_DIR": self._lapacke_dir(spec),
            "cmake.define.LAPACK_LIBS": " ".join(lapack_ldflags(spec)),
            "cmake.define.PARCEL_EXTRA_FLAGS": " ".join(rpath_flags(spec)),
        }

    @run_after("install")
    def check_native_layer(self):
        found = find(self.prefix, "libparcel_py.so")
        if not found:
            raise InstallError("libparcel_py.so が入っていない")

    # ---- spack test run py-parcel-sycl (インストール後の試験) ----
    #   examples/quick_start.py (1 次元ラプラシアン n=1000 を pcg で解き, 厳密解との最大相対誤差が 1e-6 以下か) を実行する。
    #   MPI ありの版は MPIEXEC (既定は MPI の mpiexec) で 2 プロセス。GPU の版は GPU の無いノードでは SKIPPED
    #   (parcel-sycl の試験と同じ。環境変数も同じ。スレッド数の既定 (計算ノード / ログインノード)・+fj +mpi の RC_ONLY=1 の既定も同じ)。
    @run_after("install")
    def copy_quick_test_sources(self):
        cache_extra_test_sources(self, ["examples/quick_start.py"])

    def test_quick_start_py(self):
        """examples/quick_start.py を実行し厳密解との誤差を判定する"""
        if self.spec.satisfies("~scipy"):
            raise SkipTest("~scipy (quick_start.py は scipy.sparse で行列を作る)")
        reason = quick_test_skip_reason(self.spec)
        if reason:
            raise SkipTest(reason)
        src = join_path(self.test_suite.current_test_cache_dir, "examples", "quick_start.py")
        quick_test_run(self.spec, quick_test_launcher(self.spec), [self.spec["python"].command.path, src], "quick_start.py")
