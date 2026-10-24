# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""PARCEL (SYCL 版) の spack レシピ: ヘッダライブラリと C / Fortran の API。"""
import os
import shutil

from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.generic import Package
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


def acpp_settings(pkg, spec):
    """ターゲット・カーネル最適化・ホスト向けの最適化のフラグを, バリアントから決める (py-parcel-sycl と共通)。"""
    if spec.satisfies("+cuda"):
        archs = spec.variants["cuda_arch"].value
        if len(archs) != 1 or archs[0] == "none":
            raise InstallError("parcel-sycl +cuda: cuda_arch を 1 つ指定する (例: cuda_arch=70)")
        target = "cuda:sm_%s" % archs[0]
    elif spec.satisfies("+rocm"):
        archs = spec.variants["amdgpu_target"].value
        if len(archs) != 1 or archs[0] == "none":
            raise InstallError("parcel-sycl +rocm: amdgpu_target を 1 つ指定する (例: amdgpu_target=gfx942)")
        target = "hip:%s" % archs[0]
    else:
        target = "omp"
    kernel = spec.variants["kernel"].value
    if kernel == "auto":
        if target.startswith("cuda:"):
            # NVIDIA の GPU: 梯子分割は付けない。V100 は平坦より遅く (公開パッケージのコード, pcg + Jacobi で 1 反復 0.85 倍)、
            #   A100・H100 (GH200) は差が無かった
            kernel = "none"
        elif target != "omp":            # AMD の GPU (hip): 梯子分割が効く (MI300A で 1 反復 0.66 倍)
            kernel = "ladder"
        elif spec.satisfies("+fj") or spec.satisfies("target=a64fx"):
            kernel = "a64fx"
        else:
            kernel = "blocking_dot"
    kernel_flags = {
        "blocking_dot": ["-D_BLOCKING_LOOP_DOT_=1"],
        # A64FX: 全カーネルのブロック化と SpMV の梯子分割 (前処理の梯子は遅くなるので 0。マニュアルの推奨オプション)
        #   ヤコビ前処理のその場計算 (PARCEL_JACOBI_ONTHEFLY) は -D_BLOCKING_LOOP_=1 からヘッダが既定 0 にするので指定しない
        "a64fx": ["-D_BLOCKING_LOOP_=1", "-DPARCEL_SPMV_LADDER=1", "-DPARCEL_PRECON_LADDER=0"],
        "ladder": ["-DPARCEL_SPMV_LADDER=1", "-DPARCEL_PRECON_LADDER=1"],
        "none": [],
    }[kernel]
    # V100 (sm_70): レジスタの上限を 255 に (既定のままだと Jacobi 前処理のその場で計算する版などが遅くなる。公開パッケージの推奨)
    if target == "cuda:sm_70":
        kernel_flags = kernel_flags + ["-Xcuda-ptxas", "-maxrregcount=255"]
    # A64FX はコード全体をホストのコンパイラ (富士通 FCC) でコンパイルする (omp.library-only)
    if kernel == "a64fx" and target == "omp":
        target = "omp.library-only"
    if spec.satisfies("+fj"):
        # 富士通コンパイラ経由のビルドで指定する。-ffp-contract=fast は積和を FMA にまとめさせる: FCC の clang モードは既定で
        #   まとめず, acpp が -Kfast の後ろに -O3 を付けるので -Kfast も効かない。コマンドラインに置くと FCC まで届く
        #   (A64FX 1 ノード, NX=256 の pcg + ヤコビ前処理で 1 反復 3 % 速い, SpMV 10 %。反復数は同じで結果の最後の桁が変わる。
        #   double-double の関数の中は Include/ddlib.hpp が FMA を禁じたまま)
        host_flags = ["-std=c++17", "-ffp-contract=fast"]
    else:
        host_flags = ["-march=native", "-mtune=native"] if spec.satisfies("+native") else []
    mpi_flags = [] if spec.satisfies("+mpi") else ["-DPARCEL_MPI=0"]
    # ks_arnoldi の直交化の既定 (set_qr_algo() を呼ばないとき)。cgs2 はライブラリの既定なので何も付けない
    #   (PS-A は mgs: 1 スレッドあたりの局所長が小さく大きなキャッシュを持つ CPU では MGS2 が速い)
    if "ks_ortho" in spec.variants and spec.variants["ks_ortho"].value == "mgs":
        kernel_flags = kernel_flags + ["-DPARCEL_KS_ORTHO_DEFAULT=mgs"]
    return target, kernel_flags, host_flags, mpi_flags


def rpath_flags(spec):
    """acpp を直接呼んでリンクするので spack のコンパイララッパの RPATH が付かない。実行時に要るライブラリの場所を埋め込む
    (LLVM の libomp.so, AdaptiveCpp の実行時ライブラリ, LAPACK)。"""
    dirs = [spec["llvm"].prefix.lib, spec["hipsycl"].prefix.lib]
    # spack で作った GCC の実行時ライブラリ (libstdc++ など) を先に探させる。AdaptiveCpp をそれで作ったとき,
    #   古い libstdc++ (OS や富士通コンパイラの場所にあるもの) が先に見つかるとリンクが GLIBCXX_3.4.26 などの未定義で失敗する
    if "gcc-runtime" in spec:
        dirs.insert(0, spec["gcc-runtime"].prefix.lib)
    if spec.satisfies("+fj"):
        dirs.append(spec["fj"].prefix.lib64)
    else:
        dirs += list(spec["lapack"].libs.directories)
    if spec.satisfies("+mpi"):
        if spec.satisfies("~fj"):
            dirs += list(spec["mpi"].libs.directories)
    return ["-Wl,-rpath," + d for d in dict.fromkeys(dirs)]


def mpicxx(spec):
    """MPI の C++ ラッパ。+fj は富士通 MPI の mpiFCC を富士通コンパイラの場所から使う
    (spack の fujitsu-mpi は外部登録ではコンパイラの条件を満たせず解決できない)。"""
    if spec.satisfies("+fj"):
        return join_path(spec["fj"].prefix.bin, "mpiFCC")
    return spec["mpi"].mpicxx


def lapack_ldflags(spec):
    """LAPACK のリンク指定。+fj は富士通 LAPACK (-lfjlapack, マニュアルの推奨オプション) を富士通コンパイラの場所から使う
    (spack の fujitsu-ssl2 は富士通コンパイラでビルドしたパッケージからしか使えず, AdaptiveCpp と組めない)。"""
    if spec.satisfies("+fj"):
        return ["-L" + spec["fj"].prefix.lib64, "-lfjlapack"]
    return spec["lapack"].libs.ld_flags.split()


def hwloc_env(spec, env):
    """LLVM の libomp.so は spack の新しい hwloc を使う。サイトの MPI が古い hwloc を先に読み込むと
    libomp.so の読み込みが hwloc_cpukinds_get_nr などの未定義で失敗し, AdaptiveCpp の CPU のバックエンドが
    使えなくなる (Wisteria-Aquarius の Open MPI 4.1.1)。spack の hwloc を先に探させる。"""
    if "hwloc" in spec:
        env.prepend_path("LD_LIBRARY_PATH", spec["hwloc"].prefix.lib)


def fj_env(spec, env):
    """+fj: AdaptiveCpp のホスト側 (omp.library-only) を富士通の FCC (clang モード) でコンパイルする設定。"""
    if spec.satisfies("+fj"):
        # 富士通のコンパイラと mpiFCC は自分のライブラリを LD_LIBRARY_PATH で探す (module fj / fjmpi が設定する)。
        #   spack のビルド環境はこの変数を消すので, 起動できるように足す
        env.prepend_path("LD_LIBRARY_PATH", spec["fj"].prefix.lib64)
        env.set("ACPP_CPU_CXX", join_path(spec["fj"].prefix.bin, "FCC"))
        env.set("ACPP_OMP_CXX_FLAGS", "-O3 -Kopenmp -Kfast -std=c++17 -Nclang")


def lapacke_include(spec):
    """lapacke.h のある場所 (OpenBLAS を外部パッケージにした場合は <prefix>/include/openblas などにある)。
    LAPACK が lapacke.h を持たない場合 (富士通 SSL2 など) は OS の OpenBLAS のヘッダを使う。"""
    #   HeaderList.directories は上位の include/ まで遡った場所を返すので, 見つかった lapacke.h そのものの場所を使う
    hdrs = [] if spec.satisfies("+fj") else [
        h for h in find_headers("lapacke", spec["lapack"].prefix.include, recursive=True)
        if os.path.basename(h) == "lapacke.h"]
    if not hdrs:
        hdrs = [join_path(d, "lapacke.h") for d in ("/usr/include/openblas", "/usr/include")
                if os.path.isfile(join_path(d, "lapacke.h"))]
    if not hdrs:
        raise InstallError("lapacke.h が見つからない")
    hdrs.sort(key=lambda h: (0 if "openblas" in h else 1, len(h)))   # OpenBLAS の LP64 版を優先する
    return os.path.dirname(hdrs[0])


def quick_test_skip_reason(spec):
    """spack test run でクイックスタートを実行できない理由 (実行できるなら None)。GPU の版は GPU の無いノード
    (ログインノードなど) では実行しない (ビルドだけを試験する)。PARCEL_TEST_RUN=0 で実行を止められる。"""
    if os.environ.get("PARCEL_TEST_RUN", "1") == "0":
        return "PARCEL_TEST_RUN=0"
    if spec.satisfies("+cuda"):
        smi = which("nvidia-smi")
        out = smi("-L", output=str, error=str, fail_on_error=False) if smi else ""
        if not smi or smi.returncode != 0 or "GPU" not in out:
            return "NVIDIA の GPU が見つからない (nvidia-smi -L)。GPU のあるノードで実行する"
    if spec.satisfies("+rocm") and not os.path.exists("/dev/kfd"):
        return "AMD の GPU が見つからない (/dev/kfd)。GPU のあるノードで実行する"
    return None


def quick_test_launcher(spec):
    """MPI ありの版の試験の起動コマンド (リスト)。examples/run_quick_checks.sh と同じ環境変数で変えられる:
    MPIEXEC (既定は MPI の bin/mpiexec)、MPIEXEC_NP_FLAG (-n)、MPI_NP (2)、MPIEXEC_ARGS (なし)。MPI なしの版は []。"""
    if not spec.satisfies("+mpi"):
        return []
    mpiexec = os.environ.get("MPIEXEC")
    extra = os.environ.get("MPIEXEC_ARGS", "").split()
    if not mpiexec:
        cand = join_path(spec["mpi"].prefix.bin, "mpiexec") if spec.satisfies("~fj") else ""
        mpiexec = cand if cand and os.path.isfile(cand) else "mpiexec"
        # Open MPI は 2 ランク以下だと各ランクを 1 コアに固定する。OMP_NUM_THREADS (quick_test_env) で複数スレッドにすると
        #   それが 1 コアに詰め込まれて遅くなるので, 固定しない (MPIEXEC・MPIEXEC_ARGS で配置を指定したときは付けない)
        if (spec.satisfies("~fj") and spec["mpi"].name == "openmpi"
                and not any(a.startswith(("--bind-to", "--map-by")) for a in extra)):
            extra = extra + ["--bind-to", "none"]
    return [mpiexec, os.environ.get("MPIEXEC_NP_FLAG", "-n"), os.environ.get("MPI_NP", "2")] + extra


def available_cpus():
    """このプロセスが使える論理 CPU 数。CPU の割り当て (affinity。ジョブの cpuset もここに出る)・cgroup v2 の CPU の上限
    (cpu.max。自分の cgroup とその上の階層)・
    ジョブのこのノードの割り当て (PBS: NCPUS、Slurm: SLURM_CPUS_ON_NODE) のうち最も小さいもの
    (examples/run_quick_checks.sh と同じ数え方)。"""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):          # macOS などは affinity を持たない
        n = os.cpu_count() or 1
    try:
        rel = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    rel = line.strip()[3:]
        # 自分の cgroup から上へたどり, 最も小さい上限を取る (ログインノードの利用者ごとの上限は上の階層
        #   (例 user.slice/user-<uid>.slice) にある)
        d = rel.strip("/") if rel is not None else None
        while d is not None:
            try:
                with open(join_path("/sys/fs/cgroup", d, "cpu.max")) as f:
                    quota, period = f.read().split()[:2]
                if quota != "max":
                    n = min(n, max(1, -(-int(quota) // int(period))))
            except (OSError, ValueError):
                pass
            d = os.path.dirname(d) if d else None
    except OSError:
        pass
    for v in ("NCPUS", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(v, "").isdigit() and int(os.environ[v]) > 0:
            n = min(n, int(os.environ[v]))
    return max(1, n)


JOB_ID_VARS = ("PBS_JOBID", "PJM_JOBID", "SLURM_JOB_ID")


def quick_test_node():
    """実行場所 ("compute" / "login") と判定の根拠。QUICK_NODE=compute|login で強制できる。それが無ければジョブの
    環境変数 (PBS: PBS_JOBID、富士通 PJM: PJM_JOBID、Slurm: SLURM_JOB_ID) があれば計算ノード, 無ければログインノード
    (examples/run_quick_checks.sh と同じ判定)。"""
    forced = os.environ.get("QUICK_NODE", "")
    if forced:
        if forced not in ("compute", "login"):
            raise InstallError("QUICK_NODE は compute か login (%s)" % forced)
        return forced, "QUICK_NODE=" + forced
    for v in JOB_ID_VARS:
        if os.environ.get(v):
            return "compute", "%s=%s" % (v, os.environ[v])
    return "login", "ジョブの環境変数 (%s) なし" % "・".join(JOB_ID_VARS)


def quick_test_env(spec, launcher):
    """試験の実行に足す環境変数 (dict) と, その説明 (表示用)。利用者が設定していれば変えない。
    OMP_NUM_THREADS: 既定は, 計算ノード (ジョブの中) では使える論理 CPU 数のすべて, ログインノードでは 4 (使える論理 CPU が
      4 未満ならその数)。MPI ありはそれをランク数で割る (最低 1。合計がすべて / 4 になる)。AdaptiveCpp の CPU (OpenMP) のバックエンドは
      OpenMP の既定のスレッド数 (omp_get_max_threads) を使うので, これで決まる。指定しないとログインノードなどでは
      全コア (例 256) のスレッドを起こし, 他の利用者と取り合って止まったようになる。
      ジョブではスケジューラが OMP_NUM_THREADS を設定していることがあり (PBS の ompthreads・ncpus, PJM の --omp), そのときはその値
    RC_ONLY: +fj +mpi (富士通の mpiexec はランクの標準出力をジョブのログに書き, ここで受け取れない) は既定 1。"""
    env, notes = {}, []
    node, why = quick_test_node()
    where = "%s (%s)" % ("計算ノード" if node == "compute" else "ログインノード", why)
    if os.environ.get("OMP_NUM_THREADS"):
        notes.append("OMP_NUM_THREADS=%s (設定済みの値)、実行場所 %s" % (os.environ["OMP_NUM_THREADS"], where))
    else:
        ncpu = available_cpus()
        nranks = int(launcher[2]) if launcher and launcher[2].isdigit() else 1
        total = ncpu if node == "compute" else min(4, ncpu)
        t = max(1, total // nranks)
        env["OMP_NUM_THREADS"] = str(t)
        how = "使える論理 CPU %d のすべて" % ncpu if node == "compute" else "4 (使える論理 CPU %d。4 未満ならその数)" % ncpu
        notes.append("OMP_NUM_THREADS=%d (既定: %s なので %s%s)"
                     % (t, where, how, "を %d ランクで割った値" % nranks if nranks > 1 else ""))
    if os.environ.get("RC_ONLY") is None and spec.satisfies("+fj") and launcher:
        env["RC_ONLY"] = "1"
        notes.append("RC_ONLY=1 (既定: 富士通の mpiexec。RC_ONLY=0 で check の行も判定する)")
    return env, notes


def quick_test_run(spec, launcher, args, label):
    """クイックスタートを実行し, 終了コード 0 と出力の check: ... PASS の行で判定する。RC_ONLY=1 なら check の行が
    無くても終了コードだけで合格 (富士通の mpiexec はランクの出力をジョブのログに出すため。+fj +mpi は既定で 1)。
    スレッド数などは quick_test_env が決め, 実行の間だけ環境変数に入れる。"""
    cmd = launcher + args
    env, notes = quick_test_env(spec, launcher)
    print("%s: %s  [%s]" % (label, " ".join(cmd), ", ".join(notes)))
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        exe = Executable(cmd[0])
        out = exe(*cmd[1:], output=str, error=str)       # 終了コードが 0 でなければ ProcessError (不合格)
        rc_only = os.environ.get("RC_ONLY", "0") == "1"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(out)
    checks = [l for l in out.splitlines() if l.startswith("check:")]
    if not checks and rc_only:
        print("%s: check の行なし。RC_ONLY=1 なので終了コード 0 で合格とする" % label)
        return
    assert checks and checks[0].endswith(" PASS"), "%s: check の PASS の行が無い" % label


class ParcelSycl(Package, CudaPackage, ROCmPackage):
    """PARCEL (Parallel Computing Elements) の SYCL 版。分散疎行列の Krylov 部分空間法
    (pcg, pbicgstab, pgmres, pcagmres) と固有値ソルバ (power, inverse_power, ks_arnoldi)。
    ライブラリ本体はヘッダだけ。+capi で C / Fortran の API (libparcel_sycl_c.so) もビルドする。"""

    homepage = "https://ccse.jaea.go.jp/software/PARCEL/"
    # 配布 tarball の URL (GitHub の Release。tools/make_tarball.py が作る parcel_sycl-<版>.tar.gz)。
    #   Release に置く前は `spack develop parcel-sycl@1.2.0 path=<配布パッケージ>` で使うか、tarball を手元の mirror に置いて使う
    #   (spack/README.md の「配布 tarball から入れる」)。
    url = "https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz"
    # 版と tarball の sha256。sha256 は tools/make_tarball.py --pin が書く (tarball に入れるレシピではこの行を
    #   sha256 なしの形に戻すので、書いた後も tarball と sha256 は変わらない。tools/make_tarball.py --check で照合)。
    #   版は Include/parcel_version.hpp の MAJOR.MINOR.PATCH と同じにする (tests/check_version.py で照合)
    version("1.2.0", sha256="d022ec88c1795daf1511f39ee7b0ca5b0c3e4ddfd930c56b6ac89b050ccf3efb")

    license("LGPL-3.0-only")

    variant("mpi", default=False, description="MPI あり (既定は MPI なしの 1 プロセス版)")
    variant("capi", default=True, description="C / Fortran の API をビルドする")
    variant("native", default=True, description="ホスト向けに -march=native -mtune=native を付ける (+fj では付けない)")
    variant("fj", default=False, description="A64FX: ホスト側を富士通コンパイラ (FCC) でコンパイルし, Fortran のモジュールを frt で作る")
    variant("kernel", default="auto", values=("auto", "blocking_dot", "a64fx", "ladder", "none"),
            description="カーネル最適化 (auto: CPU は blocking_dot, NVIDIA の GPU (+cuda) は none, AMD の GPU (+rocm) は ladder, A64FX は a64fx)")
    variant("ks_ortho", default="cgs2", values=("cgs2", "mgs"),
            description="ks_arnoldi の直交化の既定 (set_qr_algo() を呼ばないとき)。mgs は -DPARCEL_KS_ORTHO_DEFAULT=mgs (PS-A で使う)")
    conflicts("+cuda", when="+rocm", msg="+cuda と +rocm は同時に指定できない")

    depends_on("c", type="build", when="+capi")
    depends_on("cxx", type="build", when="+capi")
    depends_on("fortran", type="build", when="+capi")
    depends_on("hipsycl", type=("build", "link", "run"))
    # AdaptiveCpp の LLVM のライブラリ (libomp.so など) は新しい形式の圧縮デバッグ情報を持ち, OS 付属の古い ld では読めない
    depends_on("binutils", type=("build", "run"), when="platform=linux")   # macOS の ld は Apple のもの (GNU の ld は Mach-O を扱えない)
    depends_on("hipsycl+cuda", when="+cuda")
    depends_on("hipsycl+rocm", when="+rocm")
    depends_on("lapack", when="~fj", type=("build", "link", "run"))
    depends_on("mpi", when="+mpi ~fj", type=("build", "link", "run"))   # +fj は富士通 MPI (mpiFCC) を直接使う
    depends_on("fj", when="+fj", type=("build", "run"))

    def setup_build_environment(self, env):
        self._acpp_env(env)

    def setup_run_environment(self, env):
        self._acpp_env(env)
        target = acpp_settings(self, self.spec)[0]
        # 利用者のプログラムをこのパッケージと同じ設定でコンパイルするための値
        env.set("PARCEL_ACPP_TARGETS", target)
        # LAPACKE のヘッダはインストール時に include/lapacke/ に写してある (計算ノードに lapacke.h が無い環境があるため)。
        #   spack load を失敗させないよう, ここでは例外を出さない
        cxxflags, ldflags = self._user_flags()
        env.set("PARCEL_CXXFLAGS", " ".join(cxxflags))
        env.set("PARCEL_MPI", "1" if self.spec.satisfies("+mpi") else "0")
        env.set("PARCEL_LDFLAGS", " ".join(ldflags))
        hwloc_env(self.spec, env)

    def _user_flags(self):
        """利用者のプログラムをこのパッケージと同じ設定でコンパイル・リンクするフラグ (PARCEL_CXXFLAGS, PARCEL_LDFLAGS)。"""
        target, kflags, hflags, mflags = acpp_settings(self, self.spec)
        lapacke_dir = join_path(self.prefix.include, "lapacke")
        if not os.path.isdir(lapacke_dir):
            try:
                lapacke_dir = lapacke_include(self.spec)
            except Exception:
                lapacke_dir = None
        cxxflags = (["--acpp-targets=" + target] + hflags + kflags + mflags
                    + ["-I" + self.prefix.include] + (["-I" + lapacke_dir] if lapacke_dir else []))
        # 利用者のプログラムのリンク用: 実行時に要るライブラリの RPATH と LAPACK
        #   (MPI のラッパ経由で acpp を呼ぶと LLVM の libomp.so などの RPATH が付かないことがあるため)
        # spack で作った GCC の libstdc++ を -l で探す場所にも先に置く (富士通の FCC などが自分で付ける -lstdc++ が
        #   古い libstdc++ を拾うと, AdaptiveCpp のライブラリが求める GLIBCXX_3.4.26 などが未定義になる)
        gcc_rt = ["-L" + self.spec["gcc-runtime"].prefix.lib] if "gcc-runtime" in self.spec else []
        ldflags = gcc_rt + rpath_flags(self.spec) + lapack_ldflags(self.spec)
        return cxxflags, ldflags

    def setup_dependent_build_environment(self, env, dependent_spec):
        self._acpp_env(env)

    def _acpp_env(self, env):
        # AdaptiveCpp が使う CUDA。ビルド時の ACPP_CUDA_PATH (例: ドライバに合わせた版) を記録したものを優先する
        rec = join_path(self.prefix, "share", "parcel", "acpp_cuda_path")
        if os.path.isfile(rec):
            env.set("ACPP_CUDA_PATH", open(rec).read().strip())
        if self.spec.satisfies("+mpi"):
            for v in ("OMPI_CXX", "MPICH_CXX", "I_MPI_CXX"):
                env.set(v, join_path(self.spec["hipsycl"].prefix.bin, "acpp"))
        fj_env(self.spec, env)

    def _cxx(self):
        acpp = join_path(self.spec["hipsycl"].prefix.bin, "acpp")
        return mpicxx(self.spec) if self.spec.satisfies("+mpi") else acpp

    def install(self, spec, prefix):
        target, kflags, hflags, mflags = acpp_settings(self, spec)
        # ヘッダ (ライブラリ本体)
        install_tree("Include", prefix.include)
        # LAPACKE のヘッダを写す (ks_arnoldi などが lapacke.h を読む。計算ノードに lapacke.h が無い環境がある)
        src_dir = lapacke_include(spec)
        mkdirp(join_path(prefix.include, "lapacke"))
        for h in ("lapacke.h", "lapacke_config.h", "lapacke_mangling.h", "lapacke_utils.h", "lapack.h"):
            if os.path.isfile(join_path(src_dir, h)):
                install(join_path(src_dir, h), join_path(prefix.include, "lapacke", h))
        mkdirp(join_path(prefix, "share", "parcel"))
        # 指定が無ければ spack の cuda (外部登録した版など) を記録する
        cuda_path = os.environ.get("ACPP_CUDA_PATH") or (spec["cuda"].prefix if spec.satisfies("+cuda") else "")
        if cuda_path:
            with open(join_path(prefix, "share", "parcel", "acpp_cuda_path"), "w") as f:
                f.write(str(cuda_path) + "\n")
        with open(join_path(prefix, "share", "parcel", "build_config.txt"), "w") as f:
            f.write("target=%s\nkernel_flags=%s\nhost_flags=%s\nmpi=%d\n"
                    % (target, " ".join(kflags), " ".join(hflags), 1 if spec.satisfies("+mpi") else 0))
        if spec.satisfies("~capi"):
            return
        # C / Fortran の API: ラッパを SYCL コンパイラで共有ライブラリにする
        mkdirp(prefix.lib)
        if spec.satisfies("+mpi"):
            for v in ("OMPI_CXX", "MPICH_CXX", "I_MPI_CXX"):
                os.environ[v] = join_path(spec["hipsycl"].prefix.bin, "acpp")
        cxx = Executable(self._cxx())
        cxx("-O3", "--acpp-targets=" + target, *hflags, *kflags, *mflags,
            "-DMAX_ITERATION=10000", "-fPIC", "-shared",
            "-I" + prefix.include, "-I" + lapacke_include(spec),
            "-o", join_path(prefix.lib, "libparcel_sycl_c.so"),
            "capi/wrapper_parcel_sycl.cpp", *lapack_ldflags(spec), *rpath_flags(spec))
        # デバッグ情報を除く。圧縮されたデバッグ情報が入ることがあり, 古い ld (Odyssey の frt が使う /usr/bin/ld 2.30 など) が
        #   「unable to initialize decompress status」でこのライブラリをリンクできない
        strip = which("strip")
        if strip and spec.satisfies("platform=linux"):   # macOS の strip は --strip-debug を受け付けない
            strip("--strip-debug", join_path(prefix.lib, "libparcel_sycl_c.so"))
        # C のヘッダ: MPI なしなら PARCEL_MPI の既定を 0 にして入れる (利用者が -DPARCEL_MPI=0 を付けなくてよい)
        install("capi/parcel_sycl_c.h", prefix.include)
        if spec.satisfies("~mpi"):
            filter_file(r"^#define PARCEL_MPI 1$", "#define PARCEL_MPI 0",
                        join_path(prefix.include, "parcel_sycl_c.h"))
        # Fortran のモジュール (.mod) と, それを含む静的ライブラリ
        # Fortran のモジュールは spack のコンパイララッパを通さずに実際のコンパイラで作る。ラッパは依存パッケージの
        #   include/ を付け足し, ROCm (llvm-amdgpu) の include/ にある flang 用の iso_c_binding.mod を gfortran が読んで止まるため
        #   +fj は富士通の frt で作る (富士通 MPI の Fortran ラッパ mpifrt の利用者のプログラムと組めるように)
        if spec.satisfies("+fj"):
            fc, pic = Executable(join_path(spec["fj"].prefix.bin, "frt")), "-KPIC"
        else:
            fc, pic = Executable(os.environ.get("SPACK_FC") or os.environ["FC"]), "-fPIC"
        with working_dir("spack-build-fortran", create=True):
            fc("-O2", pic, "-c", "../capi/parcel_sycl_mod.f90", "-o", "parcel_sycl_mod.o")
            ar = which("ar")
            ar("rcs", join_path(prefix.lib, "libparcel_sycl_f.a"), "parcel_sycl_mod.o")
            install("parcel_sycl_mod.mod", prefix.include)
        install("capi/parcel_sycl_mod.f90", prefix.include)

    @run_after("install")
    def check_install(self):
        if self.spec.satisfies("+capi"):
            lib = join_path(self.prefix.lib, "libparcel_sycl_c.so")
            if not os.path.isfile(lib):
                raise InstallError("libparcel_sycl_c.so ができていない")

    # ---- spack test run parcel-sycl (インストール後の試験) ----
    #   インストール先のヘッダ (と +capi なら libparcel_sycl_c.so・Fortran のモジュール) でクイックスタートをビルドして実行し,
    #   1 次元ラプラシアン (n=1000) の厳密解との最大相対誤差が 1e-6 以下かを見る (examples/run_quick_checks.sh と同じ判定)。
    #   MPI ありの版は MPIEXEC (既定は MPI の mpiexec) で 2 プロセス。GPU の版は GPU の無いノードではビルドだけを試験し,
    #   実行は SKIPPED になる (GPU のあるノードで spack test run を実行する)。
    #   スレッド数は OMP_NUM_THREADS が無ければ, 計算ノード (ジョブの中) では使える論理 CPU 数のすべて, ログインノードでは
    #   4 (MPI ありは合計がそれになるようランク数で割る。QUICK_NODE=compute|login で強制)。+fj +mpi は RC_ONLY=1 が既定 (quick_test_env)。
    quick_test_sources = ["examples/quick_start.cpp", "examples/quick_start_c.c", "examples/quick_start_f.f90",
                          "examples/sample_f_mpi_stub.f90"]

    @run_after("install")
    def copy_quick_test_sources(self):
        cache_extra_test_sources(self, self.quick_test_sources)

    def _test_src(self, name):
        return join_path(self.test_suite.current_test_cache_dir, "examples", name)

    def _test_compiler(self, lang):
        """C ("c") / Fortran ("fortran") のコンパイラ。MPI ありは MPI のラッパ (中身はこのパッケージを作ったコンパイラ)。"""
        try:
            pkg = self.spec[lang].package
            real = pkg.cc if lang == "c" else pkg.fortran
        except Exception:
            real = None
        if not self.spec.satisfies("+mpi"):
            return real or which("gcc" if lang == "c" else "gfortran").path
        mpi = self.spec["mpi"]
        wrapper = getattr(mpi, "mpicc" if lang == "c" else "mpifc")
        if real:   # MPI のラッパが呼ぶコンパイラ (OS の古い gfortran がインストールしたモジュールを読めないことがある)
            for v in (("OMPI_CC", "MPICH_CC") if lang == "c" else ("OMPI_FC", "MPICH_FC")):
                os.environ[v] = real
            if mpi.name in ("intel-oneapi-mpi", "intel-mpi"):
                # Intel MPI: spack の mpicc / mpifc は mpiicx / mpiifx (中身は icx / ifx)。ifx は gfortran で作った
                #   parcel_sycl_mod.mod を読めない。mpiifx は I_MPI_F90 で gfortran に替えても ifx 用の mpi.mod (include/mpi) を
                #   読ませるので, 同じ場所の汎用のラッパ (mpicc / mpif90。mpif90 は gfortran の版に合う mpi.mod を選ぶ) を使い,
                #   中身を I_MPI_CC / I_MPI_F90 (と I_MPI_FC) で spack のコンパイラに替える
                for v in (("I_MPI_CC",) if lang == "c" else ("I_MPI_F90", "I_MPI_FC")):
                    os.environ[v] = real
                generic = join_path(os.path.dirname(wrapper), "mpicc" if lang == "c" else "mpif90")
                if os.path.isfile(generic):
                    wrapper = generic
                # 汎用のラッパは I_MPI_ROOT で自分の場所 (include/ と include/gfortran/<版>/mpi.mod) を知る。spack test の
                #   環境には無いことがあるので (外部登録の intel-oneapi-mpi)、ラッパの bin/ の 1 つ上を与える
                root = os.path.dirname(os.path.dirname(wrapper))
                if not os.environ.get("I_MPI_ROOT") and os.path.isdir(join_path(root, "include")):
                    os.environ["I_MPI_ROOT"] = root
        return wrapper

    def test_quick_start_cxx(self):
        """C++ のクイックスタート (examples/quick_start.cpp) をインストール先のヘッダでビルドして実行する"""
        if self.spec.satisfies("+mpi"):
            for v in ("OMPI_CXX", "MPICH_CXX", "I_MPI_CXX"):
                os.environ[v] = join_path(self.spec["hipsycl"].prefix.bin, "acpp")
        cxxflags, ldflags = self._user_flags()
        with test_part(self, "test_quick_start_cxx_build", purpose="quick_start.cpp をビルドする"):
            # quick_start.cpp は "../Include/parcel_sycl.hpp" を読むので, 作業場所に examples/ と
            #   Include (インストール先の include/ へのリンク) を並べてコンパイルする
            mkdirp("examples")
            shutil.copyfile(self._test_src("quick_start.cpp"), join_path("examples", "quick_start.cpp"))
            if not os.path.lexists("Include"):
                os.symlink(self.prefix.include, "Include")
            Executable(self._cxx())(*cxxflags, "-O3", "-o", "quick_start", join_path("examples", "quick_start.cpp"),
                                    *ldflags)
        with test_part(self, "test_quick_start_cxx_run", purpose="quick_start を実行し厳密解との誤差を判定する"):
            reason = quick_test_skip_reason(self.spec)
            if reason:
                raise SkipTest(reason)
            quick_test_run(self.spec, quick_test_launcher(self.spec), ["./quick_start"], "quick_start")

    def test_quick_start_c(self):
        """C のクイックスタート (examples/quick_start_c.c) を libparcel_sycl_c.so でビルドして実行する"""
        if self.spec.satisfies("~capi"):
            raise SkipTest("~capi (C / Fortran の API を入れていない)")
        if self.spec.satisfies("+fj"):
            raise SkipTest("+fj の C の試験は用意していない (spack/stack/check_stack.sh を使う)")
        lib = self.prefix.lib
        with test_part(self, "test_quick_start_c_build", purpose="quick_start_c.c をビルドする"):
            Executable(self._test_compiler("c"))("-O2", "-I" + self.prefix.include, "-o", "quick_start_c",
                                                 self._test_src("quick_start_c.c"), "-L" + lib, "-lparcel_sycl_c",
                                                 "-Wl,-rpath," + lib, "-lm")
        with test_part(self, "test_quick_start_c_run", purpose="quick_start_c を実行し厳密解との誤差を判定する"):
            reason = quick_test_skip_reason(self.spec)
            if reason:
                raise SkipTest(reason)
            quick_test_run(self.spec, quick_test_launcher(self.spec), ["./quick_start_c"], "quick_start_c")

    def test_quick_start_f(self):
        """Fortran のクイックスタート (examples/quick_start_f.f90) を parcel_sycl_mod でビルドして実行する"""
        if self.spec.satisfies("~capi"):
            raise SkipTest("~capi (C / Fortran の API を入れていない)")
        if self.spec.satisfies("+fj"):
            raise SkipTest("+fj の Fortran の試験は用意していない (spack/stack/check_stack.sh を使う)")
        lib = self.prefix.lib
        # MPI なしは "use mpi" を 1 プロセス用の代用モジュール (sample_f_mpi_stub.f90) で満たす (先にコンパイルして mpi.mod を作る)
        stub = [] if self.spec.satisfies("+mpi") else [self._test_src("sample_f_mpi_stub.f90")]
        with test_part(self, "test_quick_start_f_build", purpose="quick_start_f.f90 をビルドする"):
            Executable(self._test_compiler("fortran"))("-O2", "-I" + self.prefix.include, "-o", "quick_start_f", *stub,
                                                       self._test_src("quick_start_f.f90"), "-L" + lib,
                                                       "-lparcel_sycl_f", "-lparcel_sycl_c", "-Wl,-rpath," + lib)
        with test_part(self, "test_quick_start_f_run", purpose="quick_start_f を実行し厳密解との誤差を判定する"):
            reason = quick_test_skip_reason(self.spec)
            if reason:
                raise SkipTest(reason)
            quick_test_run(self.spec, quick_test_launcher(self.spec), ["./quick_start_f"], "quick_start_f")
