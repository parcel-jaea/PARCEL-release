#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""回帰試験・性能試験の生成器が使う「環境ごとの設定」。

  tests/regression/gen_regression.py と tests/perf/gen_perf.py (gen_perf_prec.py) が読む。
  自分の環境で試験を動かすときは、このファイルだけを書き換えればよい。

ここに並んでいる 8 環境は、開発時に試験を行った計算機の設定を**例**として残したもの。
  s86_cpu / s86_gpu   PBS、Xeon Gold 6242R 20 コア x2 / V100-SXM2-32GB x4
  wis_ody / wis_aqu   富士通 TCS (pjsub)、A64FX (Wisteria-Odyssey) / A100-SXM4-40GB (Wisteria-Aquarius)
  miy_c / miy_g       PBS (-W group_list)、Xeon MAX 9480 x2 (Miyabi-C) / GH200 (Miyabi-G)
  ps_a / ps_b         PBS (-P)、Xeon 6980P x2 / MI300A x4 (Plasma Simulator)
  mac                 スケジューラ無し (tests/regression/run_local.sh で流す)、Apple M4 10 コア (macOS + Homebrew)。回帰試験だけ
  s86spack_cpu / s86spack_gpu / miy_cspack / miy_gspack / ps_aspack / ps_bspack / wis_aquspack / wis_odyspack
                      s86 / Miyabi-C / Miyabi-G / PS-A / PS-B / Wisteria-Aquarius / Wisteria-Odyssey を、ゼロから入れた spack 環境
                      (spack/stack/install_stack.sh) で使い、spack に入れた PARCEL (parcel-sycl / py-parcel-sycl) そのものを
                      試す例 (parcel_from="installed")

書き換える場所:
  PROJECT         ジョブの課金先 (#PBS -P / #PBS -W group_list / #PJM -g)。YOUR_* はプレースホルダ
  SPACK_ROOT      Spack の場所 (AdaptiveCpp を Spack で入れている環境)
  ENV_SETUP       ジョブの先頭で流すモジュール・環境変数の設定 (CXX と WORKDIR を必ず決める)
  REGRESSION      回帰試験のキュー・資源指定・コンパイルフラグ・起動コマンド (sched="local" はスケジューラの無い計算機)
  REGRESSION_NOMPI  MPI なしの構成 (-DPARCEL_MPI=0) も試験する環境
  REGRESSION_LANG   C / Fortran の API の試験 (examples/sample_c.c / sample_f.f90) も回帰に載せる環境
  REGRESSION_LANG_PY  Python の API の試験 (tests/regression/sample_py.py) も回帰に載せる環境
  SPACK_STACK     spack/stack/install_stack.sh の入れ先 (s86spack_* が使う。ジョブの PARCEL_STACK が優先)
  SPACK_STACK_SITE  同じく他の計算機の入れ先 (miy_cspack などが使う)
  PERF            性能試験のキュー・資源指定・並列配置
使わない環境は REGRESSION / PERF から消してよい (生成器は並んでいる環境のぶんだけ作る)。

ENV_SETUP のシェル断片は次を満たすこと:
  - CXX にコンパイラ (MPI ラッパ) を入れる。追加のリンクオプションは LINKOPT に入れてよい
    (MPI なしの構成では行頭の CXX= の行を NOMPI_CXX に置き換えるので、CXX= は 1 行で書く)
  - LINKOPT に LLVM の lib/ (libomp.so の場所) の RPATH (-Wl,-rpath,$llvm_root/lib) を入れる。AdaptiveCpp は
    acpp を直接呼んでも MPI のラッパ経由でも、自分のライブラリ (libacpp-rt など) の場所しか埋め込まない
    (CPU のバックエンドが使う libomp.so の場所は埋め込まない)。無いと、ジョブの LD_LIBRARY_PATH が無いときに
    OS の libomp (または見つからない) になる。LLVM が OS のもの (/usr) なら要らない (例: miyc)
  - WORKDIR にパッケージのルート (qsub / pjsub を実行したディレクトリ) を入れる
"""

# ---------------------------------------------------------------- 課金先・パス
# ジョブの課金先。PBS の -P、Miyabi の -W group_list、富士通 TCS の -g に入る。
PROJECT = {
    "s86":      "YOUR_PROJECT",
    "ps":       "YOUR_PROJECT",
    "miyabi_c": "YOUR_GROUP",
    "miyabi_g": "YOUR_GROUP",
    "wisteria": "YOUR_GROUP",
}

# Spack のインストール先 (Wisteria / Miyabi のように $HOME 以外に置く場合)
SPACK_ROOT = "/work/YOUR_GROUP/YOUR_USER/spack"

# ゼロから入れた spack 環境 (spack/stack/install_stack.sh <環境> <入れ先>) の入れ先。s86spack_* の環境が使う
#   (ジョブに PARCEL_STACK を与えればそちらが優先)。シェルの中で展開される
SPACK_STACK = "$HOME/parcel_stack"
# 他の計算機の入れ先 (install_stack.sh の環境名ごと)。miy_cspack などが使う (ジョブの PARCEL_STACK が優先)
SPACK_STACK_SITE = {
    "miyc":    "/work/YOUR_GROUP/YOUR_USER/parcel_stack_miyc",
    "miyg":    "/work/YOUR_GROUP/YOUR_USER/parcel_stack_miyg",
    "psa":     "$HOME/parcel_stack_psa",
    "psb":     "$HOME/parcel_stack_psb",
    "wis_aqu": "/work/YOUR_GROUP/YOUR_USER/parcel_stack_wis_aqu",
    "wis_ody": "/work/YOUR_GROUP/YOUR_USER/parcel_stack_wis_ody",
}

# s86 の GPU ビルドで使う CUDA
S86_CUDA_MODULE = "cuda/12.6"
S86_CUDA_ROOT = "/home/app/cuda/12.6"

# PS の計算ノードには OpenBLAS のライブラリはあるが lapacke.h が無い。
#   nvhpc が素の lapacke.h を持っているので、ヘッダだけそこから借り、実装は OpenBLAS に解決させる。
PS_LAPACKE = ("-I/system/apps/rhel9/dev/nvhpc/25.3/Linux_x86_64/25.3/compilers/include/lp64 "
              "-l:libopenblaso.so.0")

# ---------------------------------------------------------------- ジョブの環境設定
_SPACK_ENV = (f"export SPACK_ROOT={SPACK_ROOT}/spack\n"
              f"export SPACK_USER_CONFIG_PATH={SPACK_ROOT}/site\n"
              f"export SPACK_USER_CACHE_PATH={SPACK_ROOT}/cache\n"
              ". $SPACK_ROOT/share/spack/setup-env.sh\n")

ENV_SETUP = {
# s86 は CPU / GPU で共通。{cuda_mod} {cuda_env} {ld} は生成器が埋める (GPU のときだけ CUDA を足す)
"s86": """. /etc/profile.d/modules.sh
module purge
module load gnu/9.5.0
module load openmpi/cur
{cuda_mod}. ~/spack/share/spack/setup-env.sh
spack env activate sycl
spack load  hipsycl
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm`
{cuda_env}export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:{ld}${{LD_LIBRARY_PATH}}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${{PBS_O_WORKDIR}}""",
# ゼロから入れた spack 環境 (bash spack/stack/install_stack.sh s86 <入れ先>) を使う s86 の例。{...} は生成器が埋める
#   ({parcel_load} は parcel_from="installed" のときだけ spack load parcel-sycl ... になる)。
#   AdaptiveCpp・LLVM・OpenBLAS は spack で作ったもの、GCC・Open MPI・CUDA は s86 の既存品
#   (spack/stack/envs/s86/spack.yaml の外部パッケージ)。入れ先はジョブの PARCEL_STACK (qsub -v PARCEL_STACK=<入れ先>)、
#   無ければ SPACK_STACK。
"s86spack": """. /etc/profile.d/modules.sh
module purge
{cuda_mod}. ${{PARCEL_STACK:-""" + SPACK_STACK + """}}/setup.sh
spack load hipsycl openblas openmpi
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm`
export openblas_root=`spack location -i openblas`
{cuda_env}export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:$openblas_root/lib/:{ld}${{LD_LIBRARY_PATH}}
{parcel_load}export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${{PBS_O_WORKDIR}}""",
"ody": """. /etc/profile.d/modules.sh
module purge
module load odyssey
""" + _SPACK_ENV + """spack env activate odyssey
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm@18`
export fj_root=/opt/FJSVxtclanga/tcsds-1.2.43
export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:$fj_root/lib64/:${LD_LIBRARY_PATH}
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
export ACPP_CPU_CXX="$fj_root/bin/FCC"
export ACPP_OMP_CXX_FLAGS="-O3 -Kopenmp -Kfast -std=c++17 -Nclang"
# A64FX の推奨の実行時の設定: OpenMP のスレッドを待ちの間も回し続ける (カーネルの起動と待ちの固定費が減る)。
#   1 ランク 1 CMG・12 スレッドの配置 (コアを取り合わない) が前提。取り合う配置では逆効果
export OMP_WAIT_POLICY=ACTIVE
export KMP_BLOCKTIME=infinite
CXX=mpiFCC
LINKOPT="-L$fj_root/lib64 -Wl,-rpath,$fj_root/lib64 -L$sycl_root/lib -Wl,-rpath,$sycl_root/lib -L$llvm_root/lib -Wl,-rpath,$llvm_root/lib"
WORKDIR=${PJM_O_WORKDIR}""",
"miyg": """. /etc/profile.d/modules.sh
module purge
module load gcc
module load cuda
module load ompi-cuda
""" + _SPACK_ENV + """spack env activate miyg
spack load  hipsycl
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm`
export cuda_root=/work/opt/local/aarch64/cores/cuda/12.6
export ACPP_CUDA_PATH=$cuda_root
export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:$cuda_root/lib64:${LD_LIBRARY_PATH}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${PBS_O_WORKDIR}""",
"miyc": """. /etc/profile.d/modules.sh
module purge
module load gcc
module load ompi
""" + _SPACK_ENV + """spack env activate miyc
spack load  hipsycl
export sycl_root=`spack location -i hipsycl`
export LD_LIBRARY_PATH=$sycl_root/lib/:${LD_LIBRARY_PATH}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
CXX=mpicxx
WORKDIR=${PBS_O_WORKDIR}""",
"psa": """. /etc/profile.d/modules.sh
module purge
module load intel
module load intelmpi
. ~/spack/share/spack/setup-env.sh
spack env activate sysa
spack load hipsycl
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm`
export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:${LD_LIBRARY_PATH}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
export I_MPI_CXX=acpp
# LAPACKE は OpenMP 版 OpenBLAS に解決するので、PARCEL 側の OpenMP と
# 入れ子にならないようスレッド数を 1 に固定する
export OPENBLAS_NUM_THREADS=1
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${PBS_O_WORKDIR}""",
"psb": """. /etc/profile.d/modules.sh
module purge
module load openmpi/5.0.7/rocm6.3.3
. ~/spack/share/spack/setup-env.sh
spack env activate sysb
spack load hipsycl
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm`
export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:${LD_LIBRARY_PATH}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
# LAPACKE は OpenMP 版 OpenBLAS に解決するので、PARCEL 側の OpenMP と
# 入れ子にならないようスレッド数を 1 に固定する
export OPENBLAS_NUM_THREADS=1
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${PBS_O_WORKDIR}""",
"aqu": """. /etc/profile.d/modules.sh
module purge
module load aquarius
module load cuda
module load ompi
""" + _SPACK_ENV + """spack env activate aquarius
export cuda_root=/work/opt/local/x86_64/cores/cuda/12.2
export sycl_root=`spack location -i hipsycl`
export llvm_root=`spack location -i llvm@18`   # llvm が複数入っている場合は版を指定する
export hwloc_root=`spack location -i hwloc`
export ACPP_CUDA_PATH=$cuda_root
export LD_LIBRARY_PATH=$hwloc_root/lib:$sycl_root/lib/:$llvm_root/lib/:$cuda_root/lib64:${LD_LIBRARY_PATH}
export KMP_STACKSIZE=1G
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
LINKOPT="-Wl,-rpath,$llvm_root/lib"
CXX=mpicxx
WORKDIR=${PJM_O_WORKDIR}
# Open MPI のセッションディレクトリと共有メモリをジョブごとに分ける。
#   /tmp をジョブ間で共有し、PID 名前空間がジョブごとに振り直される環境では、同時に始まった
#   2 ジョブの mpirun が同じ /tmp/ompi.<host>.<uid>/pid.<PID>/ を奪い合って起動に失敗する。
#   ログインノードでのビルドでは PJM_JOBID が無いのでシェルの PID を使う。
PARCEL_SESS=/tmp/parcel_${PJM_JOBID:-$$}
mkdir -p ${PARCEL_SESS}
export OMPI_MCA_orte_tmpdir_base=${PARCEL_SESS}
export OMPI_MCA_btl_vader_backing_directory=${PARCEL_SESS}
trap 'rm -rf ${PARCEL_SESS}' EXIT""",
# macOS + Homebrew (スケジューラ無し)。PARCEL_BREW で Homebrew の場所を変えられる。
#   macOS の /bin/bash は 3.2 なので、ここは bash 3.2 で動く書き方にする。
"mac": """BREW=${PARCEL_BREW:-/opt/homebrew}
export PATH=${BREW}/bin:${PATH}
export MPICXX_CXX=acpp
export OMPI_CXX=acpp
# LAPACKE (参照解) は OpenBLAS。PARCEL の OpenMP と入れ子にならないようスレッド数を 1 に固定する
export OPENBLAS_NUM_THREADS=1
# macOS には GNU coreutils の timeout が無いので、同じ呼び方 (timeout [-k <猶予秒>] <秒> <コマンド>...) で
#   打ち切ったときに 124 を返す関数を perl で用意する (子に TERM、猶予の後に KILL を送る)。
timeout () {
  perl -e '
    my $k = 0; if ($ARGV[0] eq "-k") { shift; $k = shift; }
    my $t = shift;
    my $pid = fork; die "fork: $!" unless defined $pid;
    if ($pid == 0) { exec { $ARGV[0] } @ARGV; exit 127; }
    my $hit = 0;
    $SIG{ALRM} = sub { $hit = 1; kill "TERM", $pid;
                       if ($k > 0) { $SIG{ALRM} = sub { kill "KILL", $pid }; alarm $k; } };
    alarm $t;
    while (waitpid($pid, 0) == -1) { last unless $!{EINTR}; }
    my $st = $?;
    exit 124 if $hit;
    exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
  ' -- "$@"
}
CXX=mpicxx
WORKDIR=${PARCEL_WORKDIR:-$(pwd)}""",
}



def _stack_env(site, cuda_root=None, extra="", workdir="${PBS_O_WORKDIR}", post=""):
    """ゼロから入れた spack 環境 (install_stack.sh <site> <入れ先>) を使うジョブの環境設定 (s86 形式以外の計算機)。
    setup.sh がサイトの module (コンパイラ・MPI・CUDA / ROCm。spack/stack/envs/<site>/site_env.sh) を読み、spack の環境 parcel を
    有効にする。AdaptiveCpp・LLVM・OpenBLAS は spack で作ったもの。{parcel_load} は生成器が埋める
    (parcel_from="installed" のとき spack load parcel-sycl ... py-parcel-sycl ... と PARCEL_PREFIX)。"""
    ld = "$sycl_root/lib/:$llvm_root/lib/:$openblas_root/lib/:"
    cuda = ""
    if cuda_root:
        cuda = f"export cuda_root={cuda_root}\nexport ACPP_CUDA_PATH=$cuda_root\n"
        ld += "$cuda_root/lib64:"
    return (". ${PARCEL_STACK:-" + SPACK_STACK_SITE[site] + "}/setup.sh\n"
            "spack load hipsycl openblas\n"
            "export sycl_root=`spack location -i hipsycl`\n"
            "export llvm_root=`spack location -i llvm`\n"
            "export openblas_root=`spack location -i openblas`\n"
            + cuda +
            "export LD_LIBRARY_PATH=" + ld + "${LD_LIBRARY_PATH}\n"
            "{parcel_load}export KMP_STACKSIZE=1G\n"
            "export MPICXX_CXX=acpp\n"
            "export OMPI_CXX=acpp\n"
            + extra +
            "LINKOPT=\"-Wl,-rpath,$llvm_root/lib\"\n"
            "CXX=mpicxx\n"
            "WORKDIR=" + workdir + post)


# LAPACKE は OpenMP 版の OpenBLAS に解決することがあるので、PARCEL の OpenMP と入れ子にならないようスレッド数を 1 に固定する
_OPENBLAS_1 = "export OPENBLAS_NUM_THREADS=1\n"
ENV_SETUP["miycspack"] = _stack_env("miyc")
ENV_SETUP["miygspack"] = _stack_env("miyg", cuda_root="/work/opt/local/aarch64/cores/cuda/12.6")
ENV_SETUP["psaspack"] = _stack_env("psa", extra="export I_MPI_CXX=acpp\n" + _OPENBLAS_1)
ENV_SETUP["psbspack"] = _stack_env("psb", extra=_OPENBLAS_1)
# Aquarius: Open MPI のセッションディレクトリをジョブごとに分ける (ENV_SETUP["aqu"] と同じ理由)
ENV_SETUP["aquspack"] = _stack_env("wis_aqu", cuda_root="/work/opt/local/x86_64/cores/cuda/12.2",
                                   workdir="${PJM_O_WORKDIR}", post="""
PARCEL_SESS=/tmp/parcel_${PJM_JOBID:-$$}
mkdir -p ${PARCEL_SESS}
export OMPI_MCA_orte_tmpdir_base=${PARCEL_SESS}
export OMPI_MCA_btl_vader_backing_directory=${PARCEL_SESS}
trap 'rm -rf ${PARCEL_SESS}' EXIT""")
# Odyssey (A64FX): ENV_SETUP["ody"] と同じく、AdaptiveCpp のホスト側を富士通の FCC (omp.library-only) で、MPI は富士通 MPI (mpiFCC)。
#   setup.sh が module odyssey を読む。spack load parcel-sycl+fj も ACPP_CPU_CXX などを同じ値にする
ENV_SETUP["odyspack"] = (". ${PARCEL_STACK:-" + SPACK_STACK_SITE["wis_ody"] + "}/setup.sh\n"
                         "spack load hipsycl\n"
                         "export sycl_root=`spack location -i hipsycl`\n"
                         "export llvm_root=`spack location -i llvm`\n"
                         "export fj_root=/opt/FJSVxtclanga/tcsds-1.2.43\n"
                         "export LD_LIBRARY_PATH=$sycl_root/lib/:$llvm_root/lib/:$fj_root/lib64/:${LD_LIBRARY_PATH}\n"
                         "{parcel_load}export MPICXX_CXX=acpp\n"
                         "export OMPI_CXX=acpp\n"
                         "export ACPP_CPU_CXX=\"$fj_root/bin/FCC\"\n"
                         "export ACPP_OMP_CXX_FLAGS=\"-O3 -Kopenmp -Kfast -std=c++17 -Nclang\"\n"
                         # A64FX の推奨の実行時の設定 (ENV_SETUP["ody"] と同じ)
                         "export OMP_WAIT_POLICY=ACTIVE\nexport KMP_BLOCKTIME=infinite\n"
                         "CXX=mpiFCC\n"
                         "LINKOPT=\"-L$fj_root/lib64 -Wl,-rpath,$fj_root/lib64 -L$sycl_root/lib -Wl,-rpath,$sycl_root/lib "
                         "-L$llvm_root/lib -Wl,-rpath,$llvm_root/lib\"\n"
                         "WORKDIR=${PJM_O_WORKDIR}")

# ---------------------------------------------------------------- 回帰試験 (tests/regression/gen_regression.py)
#
#   sched       pbs (s86 形式: -P) / pbs_miy (-W group_list) / pbs_ps (-P) / tcs (富士通 pjsub) /
#               local (スケジューラ無し。ジョブスクリプトに資源の指定を書かず、tests/regression/run_local.sh が
#               bash で流す。project・queue・select は使わない)
#   project     PROJECT のキー
#   queue, walltime, select   ジョブの資源。np ごとのジョブでは per_rank から組み立て直す
#   per_rank    1 ランクあたりの資源。ncpus = スレッド数、chunk=True は 1 ランク 1 チャンク、
#               node_ncpus はノード全体を取る (ランクを NUMA に固定するため)、tcs は TCS の書式
#   target      SYCL のターゲットと最適化フラグ。extra は環境専用の -D
#   lapack      LAPACKE のインクルードとリンク (None は PS_LAPACKE)
#   omp         1 ランクあたりのスレッド数
#   launch      起動コマンド (${NP} は並列数、${WORKDIR}/gpu_bind_TAG.sh は GPU の割り当て)
#   env         ENV_SETUP のキー。gpu は GPU 環境か
#   mpi_impl    intel なら Intel MPI (--map-by を付けず、pin の方法で固定する)
#   pin         ランクを NUMA に固定するシェル関数 (gen_regression.py の PIN_SETUP)
#   build_par   ビルドジョブの並列数 (既定は精度の数)
#   build_queue / build_walltime   ビルドジョブだけ別のキュー・制限時間にする場合
#   place       PBS の配置 (#PBS -l place=...)。(1 ノードに収まるとき, 収まらないとき) の組で、チャンクの数が node_chunks
#               (1 ノードに載るチャンクの数) 以下なら前者を使う。無ければ指定しない (キューの既定)。
#               node_chunks を与えた環境では、np が node_chunks を超えると起動コマンドの 1 ノードのランク数 (ppr) を
#               node_chunks にする (ceil(np / node_chunks) ノードに詰める)
#   map_by      False なら起動コマンドに --map-by / --bind-to を足さない (プロセスをコアに固定できない macOS など)
#   lang_fc_rt  C / Fortran の試験のリンクで足す gfortran の実行時ライブラリ (既定は Linux の libgfortran.so を探す形)
#   rank_out    起動コマンドがランクの標準出力を書くファイル (例: 富士通の mpiexec -std <ファイル>)。実行後に result.txt に足す
#               (富士通の mpiexec はランクの標準出力を、シェルのリダイレクト先ではなくジョブのログに書く)
#   py_check_launch  MPI ありの言語別ビルドで、Python から parcel を読めるかの確認に前置する起動コマンド
#               (富士通 MPI は mpiexec を通さないと MPI を初期化できない。既定は無し)
#   lang_fmod   C / Fortran の試験で Fortran のモジュール (.mod) の出力先を指定するオプション (既定 "-J"。富士通の frt は "-M ")
#   ladder      GPU の環境で False にすると梯子分割 (-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1) を付けない (既定は付ける)。
#               NVIDIA の GPU (V100・A100・H100) は False (V100 は平坦より遅く、A100・H100 は差が無かった)。AMD の MI300A は付ける
#   py_device   Python の試験で、デバイス名に含まれるべき文字列 ("|" で候補を並べる)。GPU の環境だけ
#   parcel_from C / Fortran / Python の試験が使う PARCEL。"source" (既定) はパッケージのソース (capi/、python/) をこの環境の
#               SYCL コンパイラで組む。"installed" はインストール済みのもの (spack の parcel-sycl / py-parcel-sycl) を使う
#               (spack_parcel にその構成を書くと ENV_SETUP の {parcel_load} で spack load する)。
#               検証サンプル (C++) と ddcheck は、source ではパッケージの Include/、installed ではインストールしたヘッダ
#               ($PARCEL_PREFIX/include) を組む
#   spack_parcel parcel_from="installed" で読み込む parcel-sycl の構成 (MPI の有無は生成器が足す)。例 "~cuda~rocm"、"+cuda"
#
#   CPU 系の 1 ランクあたりのスレッド数は環境ごとに固定し、np = 1/2/4 で資源を増やす
#   (s86 5 / A64FX 12 (1 CMG) / Miyabi-C 7 (1 NUMA に 2 ランク) / PS-A 7)。
#   CPU のブロックヤコビの分割数はランクあたりスレッド数なので、スレッド数を変えると分割も変わる。
REGRESSION = {
    "s86_cpu": dict(
        sched="pbs", project="s86", queue="sc8d", walltime="01:00:00",
        select="select=1:ncpus=40:mpiprocs=4:ompthreads=5",
        target="--acpp-targets=omp -mlong-double-128 -march=native -mtune=native",
        extra="", lapack="-I/usr/include/openblas -lopenblas",
        per_rank=dict(ncpus=5, ompthreads=5),
        omp="5", launch="mpirun -n ${NP}", env="s86", gpu=False, prebuilt=True),
    "s86_gpu": dict(
        sched="pbs", project="s86", queue="sg8d", walltime="01:00:00",
        select="select=1:ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4",
        target="--acpp-targets=cuda:sm_70 -march=native -mtune=native",
        # V100 (sm_70) では ptxas のレジスタ上限を広げる (-Xcuda-ptxas -maxrregcount=255)。既定の割り当てでは
        #   Jacobi 前処理などのカーネルで読み出しが分かれて遅くなる。spill は出ず、結果は変わらない。
        #   他の GPU には付けていない (効果は GPU の世代による)。
        extra="-Xcuda-ptxas -maxrregcount=255", lapack="-I/usr/include/openblas -lopenblas",
        # NVIDIA の GPU には梯子分割 (-DPARCEL_SPMV_LADDER / -DPARCEL_PRECON_LADDER) を付けない (V100 は平坦より遅かった)
        ladder=False,
        per_rank=dict(ncpus=12, ompthreads=12, ngpus=1),
        omp="12", launch="mpirun -n ${NP} ${WORKDIR}/gpu_bind_TAG.sh", env="s86", gpu=True, prebuilt=True,
        py_device="V100"),
    # A64FX はビルドが重い (1 本 6〜14 分、ノードのメモリ 28 GiB) ので、ビルドジョブは
    #   計算ノードで長めのキューを使い、並列数を抑える。
    "wis_ody": dict(
        sched="tcs", project="wisteria", queue="debug-o", walltime="0:30:00",
        select="node=1", mpi="proc=4", thread="12",
        target="--acpp-targets=omp.library-only -std=c++17",
        # A64FX 専用。-ffp-contract=fast は積和を FMA にまとめさせる (富士通コンパイラの clang モードは既定でまとめず、
        #   acpp が -Kfast の後ろに -O3 を付けるので -Kfast も効かない。README の「ビルド」)。wis_odyspack も同じ
        extra="-D_BLOCKING_LOOP_=1 -ffp-contract=fast",
        lapack="-I/usr/include/openblas -lfjlapack",
        per_rank=dict(ompthreads=12, tcs="node=1"),   # 1 ノードに 4 ランクまで載る
        # 富士通の mpiexec はランクの標準出力をジョブのログに書く (> result.txt に入らない) ので、-std でファイルに書かせて
        #   result.txt に足す (rank_out)。検証サンプルは自分で結果ファイルを書くので合否には効かないが、result.txt を残すため
        omp="12", launch="mpiexec -n ${NP} -std mpiexec_out.txt", rank_out="mpiexec_out.txt", env="ody", gpu=False,
        prebuilt=True, build_par=8, build_queue="short-o", build_walltime="2:00:00"),
    "wis_aqu": dict(
        sched="tcs", project="wisteria", queue="share-debug", walltime="0:30:00",
        select="gpu=4", mpi="proc=4", thread="9",
        target="--acpp-targets=cuda:sm_80",
        extra="", lapack="-I/usr/include/openblas -lopenblas",
        ladder=False,                                   # NVIDIA の GPU には梯子分割を付けない (A100 は差が無かった)
        per_rank=dict(ompthreads=9, tcs="gpu={np}"),   # 1 ランク 1 GPU
        omp="9", launch="mpirun -n ${NP} ${WORKDIR}/gpu_bind_TAG.sh", env="aqu", gpu=True, prebuilt=True),
    # GH200 を MIG で 4 分割したインスタンス (GPU の 1/4 + 18 コア) を 1 ランクに 1 つ割り当てる。
    "miy_g": dict(
        sched="pbs_miy", project="miyabi_g", queue="short-mig", walltime="04:00:00",
        select="select=4:mpiprocs=4:ompthreads=18",
        target="--acpp-targets=cuda:sm_90 -march=native -mtune=native",
        extra="", lapack="-I/usr/include/openblas -lopenblas",
        ladder=False,                                   # NVIDIA の GPU には梯子分割を付けない (H100 は差が無かった)
        per_rank=dict(ncpus=18, ompthreads=18, chunk=True),
        omp="18", launch="mpirun -n ${NP} ${WORKDIR}/gpu_bind_TAG.sh",
        env="miyg", gpu=True, prebuilt=True, py_device="H100|GH200"),
    # Intel MPI。1 ランク 7 スレッドを NUMA 1 つに固定する (I_MPI_PIN_DOMAIN)。
    "ps_a": dict(
        sched="pbs_ps", project="ps", queue="A_S", walltime="00:30:00",
        select="select=1:ncpus=256:mpiprocs=4:ompthreads=7",
        target="--acpp-targets=omp -march=native -mtune=native",
        # ks_arnoldi の直交化の既定を MGS2 に (PS-A では MGS2 が速い。spack の parcel-sycl ks_ortho=mgs と同じ)
        extra="-DPARCEL_KS_ORTHO_DEFAULT=mgs", lapack=None,          # PS_LAPACKE に解決する
        per_rank=dict(ncpus=7, ompthreads=7, node_ncpus=256),
        omp="7", launch="mpirun -n ${NP} -ppn ${NP}", env="psa", gpu=False, prebuilt=True,
        mpi_impl="intel", pin="psa_numa"),
    # MI300A は 1 NUMA ノード = 1 vnode = 1 APU (24 コア / 1 GPU)。GPU を 4 枚使うには chunk を 4 つ取る。
    #   -ffast-math は double-double と NaN の検査を壊すので付けない (ライブラリも拒否する)。
    "ps_b": dict(
        sched="pbs_ps", project="ps", queue="B_S", walltime="00:30:00",
        select="select=4:ncpus=24:mpiprocs=1:ompthreads=6:ngpus=1",
        target='--acpp-targets=hip:gfx942 -DNDEBUG -fno-math-errno -fno-trapping-math',
        extra="", lapack=None,          # PS_LAPACKE に解決する
        per_rank=dict(ncpus=24, ompthreads=6, ngpus=1, chunk=True),
        omp="6", launch="mpirun -n ${NP} ${WORKDIR}/gpu_bind_TAG.sh",
        env="psb", gpu=True, prebuilt=True, py_device="MI300|gfx942",
        # GPU の割り当て (gpu_bind) はノード内の順番 (ローカルランク) で GPU を選ぶ。チャンクが空いた複数のノードに
        #   散らばると、ジョブに割り当てられていない (他のジョブの) GPU を使い、遅くなる (np=4 が 3 ノードに分かれて
        #   打ち切りになった例がある)。ノードをまたぐと縮約の順序も変わり、低精度の構成では反復数が変わる。
        #   np が 1 ノード (4 APU) 以下なら 1 ノードに詰め (pack)、超えるならホストを専有して ceil(np/4) ノードに 4 ランクずつ
        #   (exclhost。gen_regression.py の place_for・launch_for)。B_S の既定 (指定なし) は free で、散らばる
        place=("pack", "exclhost"), node_chunks=4),
    "miy_c": dict(
        sched="pbs_miy", project="miyabi_c", queue="short-c", walltime="04:00:00",
        select="select=1:mpiprocs=4:ompthreads=7",
        target="--acpp-targets=omp -march=native -mtune=native",
        extra="", lapack="-I/usr/include/openblas -lopenblas",
        per_rank=dict(ncpus=7, ompthreads=7),
        omp="7", launch="mpirun -n ${NP}", env="miyc", gpu=False, prebuilt=True),
    # macOS + Homebrew (adaptivecpp・open-mpi・openblas・gcc・python)。スケジューラ無し (run_local.sh で流す)。
    #   1 ランク 5 スレッド (s86_cpu と同じ)。CPU のブロックヤコビのブロック数はランク数 x スレッド数で決まるので、
    #   他の環境と同じ分け方にして結果を照合できるようにする (np=4 で 20 スレッドとコア数を超えるが、回帰試験の
    #   問題は小さく、run_local.sh はコア数を超えるジョブを単独で流す)。Open MPI は macOS でプロセスを
    #   コアに固定できないので --bind-to none。-mlong-double-128 (x86 専用) は付けない。
    #   Homebrew の AdaptiveCpp は omp でもコード全体をホストのコンパイラ (Apple clang) で組む形 (library-only)。
    "mac": dict(
        sched="local", queue="-", walltime="-", select="-",
        target="--acpp-targets=omp -march=native -mtune=native",
        extra="",
        lapack="-I${BREW}/opt/openblas/include -L${BREW}/opt/openblas/lib -lopenblas",
        omp="5", launch="mpirun -n ${NP} --bind-to none", map_by=False,
        env="mac", gpu=False, prebuilt=True, build_par=4,
        lang_fc_rt="-L${BREW}/opt/gcc/lib/gcc/current -lgfortran -Wl,-rpath,${BREW}/opt/gcc/lib/gcc/current"),
}

# s86 の 2 環境を、ゼロから入れた spack 環境 (ENV_SETUP の s86spack) で流す例。キュー・資源・フラグは s86_cpu / s86_gpu と同じで、
#   コンパイラ・LAPACK (OpenBLAS) を spack のものにし、spack に入れた PARCEL (parcel-sycl / py-parcel-sycl) を試す
#   (parcel_from="installed"、spack_parcel はその構成)。パッケージのソースを spack のツールチェーンで組むなら parcel_from="source"。
_SPACK_LAPACK = "-I${openblas_root}/include -L${openblas_root}/lib -lopenblas -Wl,-rpath,${openblas_root}/lib"
REGRESSION["s86spack_cpu"] = dict(REGRESSION["s86_cpu"], env="s86spack", lapack=_SPACK_LAPACK,
                                  parcel_from="installed", spack_parcel="~cuda~rocm")
REGRESSION["s86spack_gpu"] = dict(REGRESSION["s86_gpu"], env="s86spack", lapack=_SPACK_LAPACK,
                                  parcel_from="installed", spack_parcel="+cuda")
# 他の計算機の spack 環境 (ENV_SETUP の miycspack など) で、spack に入れた PARCEL そのものを試す例。キュー・資源・フラグは
#   元の環境と同じで、コンパイラ・LAPACK (OpenBLAS) を spack のものにし、C / Fortran / Python は parcel-sycl / py-parcel-sycl を、
#   検証サンプル (C++) はそのヘッダを使う (parcel_from="installed")
for _plat, _env, _parcel in (("miy_c", "miycspack", "~cuda~rocm"), ("miy_g", "miygspack", "+cuda"),
                             ("ps_a", "psaspack", " ks_ortho=mgs ~cuda~rocm"), ("ps_b", "psbspack", "+rocm"),
                             ("wis_aqu", "aquspack", "+cuda")):
    REGRESSION[_plat + "spack"] = dict(REGRESSION[_plat], env=_env, lapack=_SPACK_LAPACK,
                                       parcel_from="installed", spack_parcel=_parcel)
REGRESSION["wis_aquspack"]["py_device"] = "A100"
# Odyssey: 富士通のコンパイラ・MPI・LAPACK で作った PARCEL (+fj)。キュー・資源・フラグは wis_ody と同じ。
#   C / Fortran の試験は富士通のコンパイラで組む (インストールした Fortran のモジュールは frt で作ってあり、gfortran では読めない)。
#   frt (と念のため fcc) は -Kopenmp を付ける: 付けずに組んだプログラムから PARCEL を複数スレッドで呼ぶと線形ソルバの結果が壊れる。
#   frt のモジュールの出力先は -M (lang_fmod)
#   富士通の mpiexec はランクの標準出力をジョブのログに書く (> result.txt に入らない) ので、-std でファイルに書かせる (rank_out)。
#   C / Fortran / Python の試験は標準出力で判定するので要る (検証サンプルと ddcheck は rank 0 が自分で結果ファイルを書く)
REGRESSION["wis_odyspack"] = dict(REGRESSION["wis_ody"], env="odyspack", parcel_from="installed", spack_parcel="+fj",
                                  launch="mpiexec -n ${NP} -std mpiexec_out.txt", rank_out="mpiexec_out.txt",
                                  py_check_launch="mpiexec -n 1 -std ${LOGDIR}/python_check_out.txt ",
                                  lang_fmod="-M ")

# ---------------------------------------------------------------- 回帰試験: MPI なしの構成
#   ここに挙げた環境ごとに、-DPARCEL_MPI=0 でビルドして mpirun / mpiexec を使わずに 1 プロセスで
#   実行する環境 <plat>_nompi を作る。設定は REGRESSION の <plat> から派生させ、変えるのは
#   コンパイラ (ENV_SETUP の CXX を NOMPI_CXX に置き換える)・-DPARCEL_MPI=0・起動コマンド (なし)・
#   並列数 (1 だけ) で、キュー・資源 (np=1 と同じ)・ターゲット・ケース・判定は元の環境と同じ。
#   実行ファイルは bin/<plat>_nompi/、結果は regression_results/<plat>_nompi/ に出る。
#   使わないなら空のリストにする。
REGRESSION_NOMPI = list(REGRESSION)
# MPI なしのビルドで使うコンパイラ (MPI のラッパではなく SYCL コンパイラそのもの)
NOMPI_CXX = "acpp"

# ---------------------------------------------------------------- 回帰試験: C / Fortran の API
#   examples/sample_c.c / sample_f.f90 (C / Fortran の API から全ソルバを呼んで独立に検証する例) を回帰試験に
#   載せる環境。ここに挙げた環境と、その MPI なしの構成 (REGRESSION_NOMPI にあれば <plat>_nompi) で
#   sub_build_lang_<plat>.sh と sub_regression_<plat>_lang_{c,f}_np<N>.sh を作る。使わないなら空のリストにする。
#   Wisteria の 2 環境 (wis_ody / wis_aqu) は開発時に C / Fortran のビルド・実行を確認していないので入れていない
#   (wis_aquspack / wis_odyspack は spack に入れた PARCEL のライブラリを使う形で入れている)。
REGRESSION_LANG = ["s86_cpu", "s86_gpu", "miy_c", "miy_g", "ps_a", "ps_b", "mac", "s86spack_cpu", "s86spack_gpu",
                   "miy_cspack", "miy_gspack", "ps_aspack", "ps_bspack", "wis_aquspack", "wis_odyspack"]
# Python の API の試験 (tests/regression/sample_py.py。sample_c と同じ問題・項目・判定を Python のパッケージ parcel から行う) を
#   回帰に載せる環境。numpy と scipy の入った Python が要る (ビルドジョブに PARCEL_REG_PY で与える。既定 python3)。
REGRESSION_LANG_PY = ["s86_cpu", "s86_gpu", "miy_c", "miy_g", "ps_a", "ps_b", "mac", "s86spack_cpu", "s86spack_gpu",
                      "miy_cspack", "miy_gspack", "ps_aspack", "ps_bspack", "wis_aquspack", "wis_odyspack"]
# C / Fortran の main のコンパイラ。MPI ありは MPI のラッパ ({plat: ...} で環境ごとに変えられる、"default" が既定)、
#   MPI なしは素のコンパイラ。リンクは ENV_SETUP の CXX (SYCL コンパイラ) で行い、Fortran の実行時ライブラリ
#   (MPI ありは MPI の Fortran ライブラリも) はラッパから調べて足す (Open MPI は --showme:link、Intel MPI は -show)。
#   MPI なしも {plat: ...} の辞書で環境ごとに変えられる ("default" が既定。文字列ならすべての環境で同じ)。
LANG_MPI_CC = {"default": "mpicc", "wis_odyspack": "mpifcc -Kopenmp"}        # 富士通 MPI
LANG_MPI_FC = {"default": "$(command -v mpifort > /dev/null 2>&1 && echo mpifort || echo mpif90)",
               "ps_a": "mpif90", "ps_aspack": "mpif90",          # Intel MPI (中身は gfortran)
               "wis_odyspack": "mpifrt -Kopenmp"}                # 富士通 MPI
LANG_NOMPI_CC = {"default": "gcc", "wis_odyspack": "fcc -Kopenmp"}
LANG_NOMPI_FC = {"default": "gfortran", "wis_odyspack": "frt -Kopenmp"}

# ---------------------------------------------------------------- 性能試験 (tests/perf/gen_perf.py, gen_perf_prec.py)
#
#   装置数 (CPU はソケット、GPU は 1 台) 4 / 8 / 16 で、1 ジョブ = 1 ソルバ x 1 装置数。
#   kind        cpu / gpu
#   rpd         1 装置あたりのランク数 (CPU: 1 ソケットのランク数、GPU: 1)。np = 装置数 x rpd
#   omp         1 ランクあたりのスレッド数
#   dpn         1 ノードあたりの装置数 (CPU: ソケット数、GPU: 台数)。ノード数 = 装置数 / dpn (GPU は切り上げ)
#   chunk       1 ノードぶんの資源指定 (select=<ノード数>:<chunk>)。chunk_per="device" なら 1 装置ぶんで、
#               select=<装置数>:<chunk> にする (PS-B のように 1 vnode = 1 装置の環境)
#   queues      [(ノード数の上限 (None は無制限), walltime の上限, キュー), ...]。上から順に、収まる最初のキューを使う
#   walltime    既定の walltime。walltime_by = {ジョブ名: {装置数: walltime}} で上書きする
#               (ジョブ名はソルバ名 pcg / pbicgstab / pgmres / pcagmres_explicit / pcagmres_implicit / power /
#               inverse_power / ks_arnoldi と、混合精度の比較 prec)。値は開発時の測定 (NX=512) の見積もりからの例
#   mpi_impl    intel なら Intel MPI (-ppn で配置し、pin の方法で固定する)
#   pin, pin_mode   ランクを NUMA に固定するシェル関数 (gen_regression.py の PIN_SETUP) と配置 (pack / spread)
#   place       PBS の配置 (#PBS -l place=...)。(1 ノードに収まるとき, 収まらないとき) の組。チャンクの数が
#               1 ノードぶん (chunk_per="device" なら dpn、それ以外は 1) 以下なら前者を使う。無ければ指定しない
#   job_args    {ジョブ名: 実行時オプション}。その環境のそのジョブの全構成に足す (例 ps_a の ks_arnoldi の
#               PARCEL_KS_QR=mgs。検証サンプルが set_qr_algo(QR_MGS) を呼ぶ)。無ければ足さない
#   その他のキーは REGRESSION と同じ意味。実行ファイルは回帰試験のビルドジョブが作る bin/<plat>/verify_<精度> を使う。
PERF = {
    "s86_cpu": dict(
        kind="cpu", sched="pbs", project="s86", env="s86", gpu=False,
        rpd=4, omp=5, dpn=2, chunk="ncpus=40:mpiprocs=8:ompthreads=5",
        queues=[(None, "24:00:00", "sc16")], walltime="01:00:00",
        walltime_by={"pgmres": {4: "02:30:00", 8: "01:30:00"},
                     "pcagmres_explicit": {4: "02:00:00", 8: "01:30:00"},
                     "pcagmres_implicit": {4: "02:00:00", 8: "01:30:00"},
                     "power": {4: "02:00:00", 8: "01:30:00"},
                     "inverse_power": {4: "10:30:00", 8: "06:00:00", 16: "03:30:00"},
                     "ks_arnoldi": {4: "03:00:00", 8: "01:30:00"},
                     "prec": {4: "01:30:00"}},
        note="Xeon Gold 6242R 20 コア x2 / ノード、1 ソケット 4 ランク x 5 スレッド"),
    "s86_gpu": dict(
        kind="gpu", sched="pbs", project="s86", env="s86", gpu=True,
        rpd=1, omp=12, dpn=4, chunk="ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4",
        queues=[(None, "01:00:00", "sg8d"), (None, "24:00:00", "sg8")], walltime="01:00:00",
        walltime_by={"inverse_power": {4: "02:30:00", 8: "01:30:00"}},
        note="V100-SXM2-32GB x4 / ノード、1 ランク 1 台 x 12 スレッド"),
    # 1 ソケット = 4 NUMA (各 14 コア)。1 NUMA に 1 ランク
    "miy_c": dict(
        kind="cpu", sched="pbs_miy", project="miyabi_c", env="miyc", gpu=False,
        rpd=4, omp=14, dpn=2, chunk="ncpus=112:mpiprocs=8:ompthreads=14",
        queues=[(2, "08:00:00", "short-c"), (None, "48:00:00", "regular-c")], walltime="01:00:00",
        walltime_by={"inverse_power": {4: "02:30:00", 8: "01:30:00"}},
        note="Xeon MAX 9480 56 コア x2 / ノード、1 ソケット 4 ランク x 14 スレッド (1 NUMA に 1 ランク)"),
    # GH200: 1 ノード = Grace 72 コア + H100 1 台。1 ランクが 1 ノードを専有する
    "miy_g": dict(
        kind="gpu", sched="pbs_miy", project="miyabi_g", env="miyg", gpu=True,
        rpd=1, omp=72, dpn=1, chunk="ncpus=72:mpiprocs=1:ompthreads=72",
        queues=[(8, "08:00:00", "short-g"), (None, "48:00:00", "regular-g")], walltime="01:00:00",
        note="GH200 (H100 96GB) 1 台 / ノード、1 ランク 1 台 x 72 スレッド"),
    # 1 ソケット = 3 NUMA。14 スレッドのランクを 1 NUMA に 3 つずつ詰めて固定する (Intel MPI)
    "ps_a": dict(
        kind="cpu", sched="pbs_ps", project="ps", env="psa", gpu=False,
        rpd=9, omp=14, dpn=2, chunk="ncpus=256:mpiprocs=18:ompthreads=14",
        mpi_impl="intel", pin="psa_numa", pin_mode="pack",
        queues=[(4, "24:00:00", "A_S"), (None, "24:00:00", "A_M")], walltime="01:00:00",
        walltime_by={"inverse_power": {4: "02:00:00", 8: "01:30:00"}},
        # ks_arnoldi の直交化は MGS2 で測る。ps_a のビルドは既定が MGS2 (REGRESSION の extra -DPARCEL_KS_ORTHO_DEFAULT=mgs) だが、
        #   実行時オプションでも明示する (検証サンプルの PARCEL_KS_QR=mgs が set_qr_algo(QR_MGS) を呼ぶ。別のフラグで作った
        #   実行ファイルでも同じ条件になり、出力の "[LIB_DEFAULTS] ... , qr = MGS (env)" で確かめられる)。1 スレッドあたりの局所長が小さく大きなキャッシュを
        #   持つこの CPU では、装置数 8・16 で MGS2 の方が速い (開発時の測定で MGS2 / CGS2 = 0.88 / 0.73、装置数 4 は 1.05)
        job_args={"ks_arnoldi": "PARCEL_KS_QR=mgs"},
        note="Xeon 6980P 128 コア x2 / ノード、1 ソケット 9 ランク x 14 スレッド (1 NUMA に 3 ランク)"),
    # MI300A は 1 vnode = 1 APU (24 コア / 1 GPU)。装置の数だけ chunk を取る
    "ps_b": dict(
        kind="gpu", sched="pbs_ps", project="ps", env="psb", gpu=True,
        rpd=1, omp=6, dpn=4, chunk="ncpus=24:mpiprocs=1:ompthreads=6:ngpus=1", chunk_per="device",
        # 装置数 4 は 1 ノードに詰め (pack)、8・16 はノードを専有する (exclhost。B_M の既定と同じ)。GPU の割り当てが
        #   ノード内の順番で GPU を選ぶので、チャンクが他のジョブと同じノードに散らばると他のジョブの GPU を使う
        #   (指定なしの B_S で装置数 8 が 1+4+3 ノードに分かれた例がある)
        place=("pack", "exclhost"),
        queues=[(2, "24:00:00", "B_S"), (None, "24:00:00", "B_M")], walltime="01:00:00",
        note="MI300A x4 / ノード (1 vnode = 1 APU)、1 ランク 1 台 x 6 スレッド"),
}
# ゼロから入れた spack 環境で測る例 (s86。実行ファイルは回帰試験の s86spack_* のビルドが作る)
PERF["s86spack_cpu"] = dict(PERF["s86_cpu"], env="s86spack")
PERF["s86spack_gpu"] = dict(PERF["s86_gpu"], env="s86spack")

# 混合精度の比較 (tests/perf/gen_perf_prec.py) の対象環境
PERF_PREC_PLATS = ["s86_cpu", "s86_gpu", "miy_c", "miy_g", "ps_a", "ps_b"]
