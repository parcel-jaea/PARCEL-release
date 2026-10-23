[English](README.md) | 日本語

# spack によるインストール

`spack/` には PARCEL_SYCL を spack で入れるためのものがある。

| 場所 | 中身 |
|---|---|
| `spack_repo/parcel/` | spack (1.x、リポジトリ API v2.2) のパッケージリポジトリ。`parcel-sycl` と `py-parcel-sycl` のレシピ |
| `stack/install_stack.sh` | spack の取得から LLVM・AdaptiveCpp・OpenBLAS・Python、PARCEL までをゼロから入れるスクリプト |
| `stack/check_stack.sh` | 入れたものの確認 (ログインノードで実行できる範囲) |
| `stack/envs/<環境>/` | 環境ごとの spack の定義 (`spack.yaml`) と追加の設定 |

## レシピ

| パッケージ | 中身 |
|---|---|
| `parcel-sycl` | ヘッダ (`include/`)、C の API `lib/libparcel_sycl_c.so` と `include/parcel_sycl_c.h`、Fortran のモジュール `include/parcel_sycl_mod.mod` と `lib/libparcel_sycl_f.a` |
| `py-parcel-sycl` | Python のパッケージ `parcel` (`python/` を scikit-build-core でビルド) |

バリアント (両方共通):

| バリアント | 既定 | 意味 |
|---|---|---|
| `mpi` | off | MPI あり (既定は MPI なしの 1 プロセス版) |
| `cuda` + `cuda_arch=<n>` | off | AdaptiveCpp の `--acpp-targets=cuda:sm_<n>`。`cuda_arch=70` (V100) では `-Xcuda-ptxas -maxrregcount=255` を自動で付ける |
| `rocm` + `amdgpu_target=<gfx>` | off | `--acpp-targets=hip:<gfx>` |
| (どちらも無し) | | `--acpp-targets=omp` (CPU) |
| `native` | on | `-march=native -mtune=native` |
| `kernel` | auto | カーネル最適化: auto (CPU は blocking_dot、NVIDIA の GPU (`+cuda`) は none、AMD の GPU (`+rocm`) は ladder、A64FX は a64fx)、blocking_dot、a64fx、ladder、none |
| `fj` | off | A64FX: ホスト側を富士通コンパイラ (FCC) でコンパイルし、Fortran のモジュールを frt で作る (Wisteria-Odyssey の定義 `envs/wis_ody` で確認)。`-std=c++17 -ffp-contract=fast` を付ける (FCC の clang モードは既定で積和を FMA にまとめない。README の「ビルド」) |
| `ks_ortho` | cgs2 | `ks_arnoldi` の直交化の既定 (プログラムが `set_qr_algo()` を呼ばないとき)。`mgs` で `-DPARCEL_KS_ORTHO_DEFAULT=mgs` を付ける (`PARCEL_CXXFLAGS` にも入る)。`envs/psa` (PS-A) は `ks_ortho=mgs` |
| `capi` (`parcel-sycl` のみ) | on | C / Fortran の API をビルドする |
| `scipy` (`py-parcel-sycl` のみ) | on | SciPy も入れる (`DistributedCSR.from_scipy` / `from_global` と `examples/quick_start.py` が使う。pip の `parcel[scipy]` に当たる) |

レシピの `url` は GitHub の Release に置く tarball (`https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz`)。
Release に置いた後は、レシピに書いた sha256 で照合して取得する (下の「配布 tarball から入れる」)。手元のソースを使うときは
`spack develop` で手元のパッケージ (このディレクトリの親) を指定する。

```bash
spack -e <環境> repo add <パッケージ>/spack/spack_repo/parcel
spack -e <環境> config add "concretizer:unify:when_possible"          # CPU 版と GPU 版などを並べるとき
spack -e <環境> develop --no-clone --path <パッケージ> parcel-sycl@1.2.0
spack -e <環境> develop --no-clone --path <パッケージ> py-parcel-sycl@1.2.0
spack -e <環境> add "parcel-sycl@1.2.0 ~mpi ~cuda ~rocm" "py-parcel-sycl@1.2.0 ~mpi ~cuda ~rocm"                    # CPU
spack -e <環境> add "parcel-sycl@1.2.0 ~mpi +cuda cuda_arch=70" "py-parcel-sycl@1.2.0 ~mpi +cuda cuda_arch=70"     # GPU
spack -e <環境> concretize -f && spack -e <環境> install

spack env activate <環境>
spack load parcel-sycl+cuda py-parcel-sycl+cuda    # 構成を指定して読む
python script.py                                   # Python
acpp $PARCEL_CXXFLAGS -O3 -o prog prog.cpp $PARCEL_LDFLAGS        # C++
P=$(spack location -i parcel-sycl+cuda)            # C: spack load はヘッダとライブラリの場所を設定しないので自分で与える
gcc -I$P/include -o prog prog.c -L$P/lib -lparcel_sycl_c -Wl,-rpath,$P/lib   # (MPI なしの版のヘッダは PARCEL_MPI の既定が 0)
```

- `spack load parcel-sycl` は、利用者のプログラムをこのパッケージと同じ設定でコンパイルするための
  `PARCEL_CXXFLAGS` (ターゲット・最適化・MPI の有無・ヘッダ) と `PARCEL_LDFLAGS` (実行時に要るライブラリ
  (LLVM の libomp.so、AdaptiveCpp、LAPACK、MPI) の RPATH と LAPACK) を設定する。MPI ありは
  `mpicxx $PARCEL_CXXFLAGS ... $PARCEL_LDFLAGS` (`OMPI_CXX=acpp` なども `spack load` が設定する)。
  `acpp` を直接呼んでも MPI のラッパ経由でも LLVM の libomp.so の RPATH は付かないので、`PARCEL_LDFLAGS` を付ける。
- ルートの仕様は `~cuda ~rocm` まで明示する。`parcel-sycl ~mpi` だけだと、unify の解決で GPU 版と同じものにまとめられる。
- CPU 版と GPU 版を同じ環境に入れるとビューで重なるので、環境のビューから `parcel-sycl`・`py-parcel-sycl` を除外する
  (`view: default: {root: .spack-env/view, exclude: [parcel-sycl, py-parcel-sycl]}`)。
- 解決済みの環境に仕様を足すと、`concretize` が内部エラー (unsatisfiable) になることがある。`concretize -f` で解決し直す。
- MPI と hipsycl は環境で 1 つにそろえる (`packages:mpi:require:` に版と構成まで、`packages:hipsycl:require:`)。
  外部登録の MPI が複数あると、別の構成 (CUDA 版など) が選ばれることがある。
- AdaptiveCpp が使う CUDA は、ビルド時の `ACPP_CUDA_PATH` (または spack の cuda) を `share/parcel/acpp_cuda_path` に記録し、
  `spack load` で同じ値を設定する。ドライバより新しい CUDA を使わないよう、外部パッケージの cuda かこの変数で指定する。
- PARCEL のヘッダは `include/`、LAPACKE のヘッダは `include/lapacke/` に写す (計算ノードに `lapacke.h` が無い環境のため)。
- **インストール後の試験** `spack test run parcel-sycl py-parcel-sycl`: インストール先のヘッダとライブラリでクイックスタート
  (`parcel-sycl`: C++、`+capi` なら C・Fortran も。`py-parcel-sycl`: `quick_start.py`) をビルドして実行し、厳密解との
  最大相対誤差が 1e-6 以下かを判定する (`examples/run_quick_checks.sh` と同じ判定)。結果は `spack test results`。
  MPI ありの版は `MPIEXEC` (既定は MPI の `bin/mpiexec`) `MPIEXEC_NP_FLAG` (`-n`) `MPI_NP` (2) `MPIEXEC_ARGS` で起動する。
  GPU の版は GPU の無いノード (ログインノード) ではビルドだけを試験し、実行は SKIPPED になるので、GPU の計算ノードでも流す。
  `PARCEL_TEST_RUN=0` で実行を止められる。`+fj` の C・Fortran は試験しない (`check_stack.sh` を使う)。
  スレッド数と実行場所の判定は `examples/run_quick_checks.sh` と同じ: `OMP_NUM_THREADS` が無ければ、計算ノード (ジョブの中) は
  使える論理 CPU 数のすべて、ログインノードは 4 (MPI ありは合計がそれになるようプロセス数で割る)。計算ノードかどうかは
  `PBS_JOBID`・`PJM_JOBID`・`SLURM_JOB_ID` の有無で決め、`QUICK_NODE=compute|login` で強制できる。判定と使った値は
  `spack test results -l` の出力に出る。`+fj +mpi` (富士通の mpiexec) は `RC_ONLY=1` が既定 (ランクの標準出力がジョブのログに出て
  check の行を読めないので、終了コードだけで判定する。外すときは `RC_ONLY=0`)。MPI が Open MPI で `MPIEXEC` を指定していなければ、
  `MPIEXEC_ARGS` に配置の指定が無いとき `--bind-to none` を付ける。Intel MPI では C・Fortran の試験を同じ場所の汎用のラッパ
  (`mpicc`・`mpif90`) で組み、中身のコンパイラを `I_MPI_CC`・`I_MPI_F90` (と `I_MPI_FC`) で spack のコンパイラ (gfortran など) にする
  (spack が返す `mpiifx` は ifx を呼び、gfortran で作った `parcel_sycl_mod.mod` を読めない。`mpiifx` は中身を gfortran にしても
  ifx 用の `mpi.mod` を読ませる)。GPU の版の実行と `+fj` (A64FX) は計算ノード (短いバッチジョブか対話ジョブ) で流す。
  CPU の版はログインノードでもよい (施設の規則に従う)。
- `spack develop` のソースの中で試験を実行すると (`bin/`、`__pycache__` などができる)、spack は更新ありとみなし、
  次の `install` で PARCEL を作り直す。試験はソースの写しで行う。

### 配布 tarball から入れる

レシピには `tools/make_tarball.py` で作った tarball (`parcel_sycl-1.2.0.tar.gz`) の sha256 が書いてある (`--pin`。
`version("1.2.0", sha256="...")`)。GitHub の Release に置いた後は、`spack develop` を使わずに `url` から取得して入れられる
(`parcel-sycl` と `py-parcel-sycl` は同じ tarball)。spack は取得した tarball をレシピの sha256 で照合する (違えば止まる)。

Release に置く前や、計算機が外部に接続できないときは、tarball を spack の mirror の形
(`_source-cache/archive/<sha256 の先頭 2 文字>/<sha256>.tar.gz`) で手元に置く。

```bash
python3 tools/make_tarball.py --outdir <出力先> --pin        # tarball と sha256、レシピ 2 本に sha256 を書く
M=<mirror の場所>
SHA=$(cut -d' ' -f1 <出力先>/parcel_sycl-1.2.0.tar.gz.sha256)
mkdir -p $M/_source-cache/archive/${SHA:0:2}
cp <出力先>/parcel_sycl-1.2.0.tar.gz $M/_source-cache/archive/${SHA:0:2}/$SHA.tar.gz
spack -e <環境> mirror add --scope env:<環境> parcel-local file://$M
spack -e <環境> repo add <パッケージ>/spack/spack_repo/parcel        # sha256 を書いたレシピ
spack -e <環境> add "parcel-sycl@1.2.0 ~mpi ~cuda ~rocm"              # develop しない
spack -e <環境> concretize -f && spack -e <環境> install
```

- tarball の中に入るレシピは sha256 なしの形 (`version("1.2.0")`) にしてある (レシピに tarball 自身の sha256 は書けないため)。
  `--pin` で手元のレシピに sha256 を書いた後に tarball を作り直しても同じ tarball になる。`--check` で照合できる。
- パッケージの中身を変えたら tarball を作り直し、`--pin` で sha256 を書き直す。公開時は、公開した tarball の sha256 を書いたレシピを使う。

### 構成の選び方

入れる構成は、パッケージとバリアントの組合せ (仕様) で選ぶ (版の `@1.2.0` は省略)。

| 目的 | 仕様 |
|---|---|
| CPU、MPI なし (C++・C・Fortran) | `parcel-sycl ~mpi ~cuda ~rocm` |
| CPU、MPI あり | `parcel-sycl +mpi ~cuda ~rocm` |
| NVIDIA の GPU (A100 の例) | `parcel-sycl ~mpi +cuda cuda_arch=80` |
| NVIDIA の GPU、MPI あり | `parcel-sycl +mpi +cuda cuda_arch=80` |
| AMD の GPU (MI300A) | `parcel-sycl ~mpi +rocm amdgpu_target=gfx942` |
| C++ (ヘッダ) だけ。C / Fortran の API を作らない | `parcel-sycl ~capi ~mpi ~cuda ~rocm` |
| Python も使う | 同じ構成の `py-parcel-sycl` も並べる (例 `py-parcel-sycl ~mpi ~cuda ~rocm`) |

- C++ (ヘッダ) は `parcel-sycl` に常に入る。C と Fortran の API は `capi` (既定 on) でまとめて入れるか外すかを選ぶ (片方だけは選べない)。
  Python は `py-parcel-sycl` を加えるかどうかで選ぶ。
- `+capi` では Fortran のコンパイラが要る。無い環境では `~capi` にする。
- `py-parcel-sycl` はビルドに `parcel-sycl` を使うので、Python だけでも `parcel-sycl` (既定の `+capi`) が入る。Fortran のコンパイラが無い環境では
  `py-parcel-sycl ~mpi ~cuda ~rocm ^parcel-sycl ~capi` のように添える (s86 で解決でき、`parcel-sycl` が Fortran のコンパイラを使わなくなることを確認)。
  OpenBLAS も既定では Fortran のコンパイラを使うので、まったく無い環境では `openblas ~fortran` も指定する (`envs/wis_ody` と同じ)。
- `+cuda` と `+rocm` は同時に指定できない。どちらも付けなければ CPU (OpenMP) 版。GPU では `cuda_arch` (V100 70、A100 80、GH200 90) か
  `amdgpu_target` を 1 つ指定する。
- `+mpi` には外部パッケージとして登録した MPI が要る。`py-parcel-sycl` の `mpi`・`cuda`・`rocm` などは組み合わせる `parcel-sycl` と同じにする。
- `stack/install_stack.sh` で入れる構成は `stack/envs/<環境>/spack.yaml` の `specs:` に並べたもの。構成を絞る・変えるときは `specs:` を書き換えてから
  `install_stack.sh` を流す (入れた後に書き換えた場合も、同じコマンドで足りないものだけを入れる)。例 (CPU・MPI ありの C++・C・Fortran と Python だけ):

```yaml
spack:
  specs:
  - parcel-sycl@1.2.0 +mpi ~cuda ~rocm
  - py-parcel-sycl@1.2.0 +mpi ~cuda ~rocm
```

## ゼロから入れる (`stack/`)

既存の spack を使わず、spack の取得から PARCEL までを 1 本のスクリプトで入れる。

```bash
bash spack/stack/install_stack.sh s86 $HOME/parcel_stack 16           # <環境> <入れ先> [並列数]
bash spack/stack/check_stack.sh $HOME/parcel_stack $HOME/parcel_check  # ログインノードでの確認
. $HOME/parcel_stack/setup.sh                                          # 使うとき
spack load parcel-sycl+mpi+cuda py-parcel-sycl+mpi+cuda
```

- spack で作るもの: LLVM 18.1.8、AdaptiveCpp (hipsycl 24.10.0)、OpenBLAS、binutils、Python 3.11 と numpy・SciPy・pip、
  PARCEL 本体・C / Fortran の API・Python のパッケージ。
- 各環境の既存品を使うもの (`stack/envs/<環境>/spack.yaml` に外部パッケージとして書く): コンパイラ、CUDA / ROCm、MPI。
  計算機のネットワークやドライバと結び付くため。
- spack 本体 (1.1.0.dev0) とレシピ集 spack-packages は、確認した版に固定する (`install_stack.sh` の先頭)。
- 入れ先の中に spack・設定・キャッシュをすべて置き、`SPACK_DISABLE_LOCAL_CONFIG=1` で `~/.spack` と `/etc/spack` を読まない
  (既存の spack の環境に影響されず、影響も与えない)。
- PARCEL のソースは `install_stack.sh` の 2 つ上のディレクトリ (このパッケージ。`Include/`、`capi/`、`python/` があること)。
  別の場所を使うときは `PARCEL_SRC=<ディレクトリ>` を与える。試験を流すディレクトリとは分けておく (上の注意)。
- `envs/<環境>/site_env.sh` は、インストール中と `setup.sh` を読んだときに読む (サイトの MPI などの module)。
  `after_activate.sh` は環境を有効にした後に、`compilers.yaml` は本体より先に入れるコンパイラ (OS の GCC が古い環境)。
- 途中で止まったら同じコマンドで再開できる。`CONCRETIZE_ONLY=1` を付けると解決だけ行う。
- `check_stack.sh` は入っている構成ごとに `spack load` し、Python から読めるか、`examples/` の `sample_c.c`・`sample_f.f90`・
  `quick_start.cpp` がビルドできて実行に要るライブラリがすべて見つかるかを調べる (実行はしない)。
  MPI なしの Fortran は `examples/sample_f_mpi_stub.f90` と一緒にコンパイルする。
- 入れた環境で回帰試験・性能試験を流す設定の例は `tests/site_config.py` の `s86spack_cpu` / `s86spack_gpu`
  (`tests/regression/README.ja.md` の「spack の環境で流す」)。

### 用意した環境

| `envs/` | 計算機 | 入れる構成 | 既存品 | 確認 |
|---|---|---|---|---|
| `s86` | x86_64 (Cascade Lake) + V100 | CPU / GPU (`cuda_arch=70`) × MPI なし / あり | GCC 9.5.0、CUDA 12.6、Open MPI 4.1.6 | `install_stack.sh` と `check_stack.sh` 合格。計算ノードで Python・C・Fortran・C++ のサンプルが合格 (レシピの最後の変更 (V100 のフラグ・Fortran のモジュール) の前の版で) |
| `miyc` | Miyabi-C (Xeon MAX 9480) | CPU × MPI なし / あり | GCC 11.4.1、Open MPI 4.1.6 | `install_stack.sh` と `check_stack.sh` 合格 |
| `miyg` | Miyabi-G (GH200) | GPU (`cuda_arch=90`) × MPI なし / あり | GCC 11.4.1、CUDA 12.6、Open MPI 4.1.6 (CUDA 版) | 同上 |
| `psa` | Plasma Simulator PS-A (Xeon 6980P) | CPU × MPI なし / あり (`ks_ortho=mgs`: `ks_arnoldi` の直交化の既定が MGS2) | GCC 11.5.0、Intel MPI 2021.18 | 同上 |
| `psb` | Plasma Simulator PS-B (MI300A) | GPU (`amdgpu_target=gfx942`) × MPI なし / あり | GCC 11.5.0、ROCm 6.3.3、Open MPI 5.0.7 (ROCm 版) | 同上 |
| `wis_aqu` | Wisteria-Aquarius (A100) | GPU (`cuda_arch=80`) × MPI なし / あり | GCC 8.3.1 (GCC 12 を作るためだけ)、CUDA 12.2、Open MPI 4.1.1 | 同上 |
| `wis_ody` | Wisteria-Odyssey (A64FX) | CPU (`+fj`) × MPI なし / あり | GCC 8.3.1 (LLVM・boost などと gfortran)、富士通のコンパイラ・MPI・LAPACK (tcsds-1.2.43) | `install_stack.sh` と `check_stack.sh` 合格。計算ノードで Python・C・Fortran・C++ のサンプルが MPI なし・あり (np=1・2・4) で合格 |
| `generic` | 上の 7 環境以外の Linux の計算機 (雛形) | CPU × MPI なし (既定)。MPI あり・NVIDIA の GPU はコメントを外して足す | 書き換えて使う (GCC、MPI、CUDA) | Ubuntu 24.04 (WSL2、GCC 13.3.0、Open MPI 4.1.6、CUDA 12.6、RTX A6000) と Rocky Linux 9.8 (WSL2、GCC 11.5.0、Open MPI 4.1.1、CUDA 12.6、RTX A6000) で CPU・GPU × MPI なし・ありを入れて確認 |

- 外部パッケージの場所 (`prefix:`) と module の名前は各計算機のシステムのもの。別の計算機では書き換える。
- Wisteria-Odyssey (`wis_ody`) はログインノードが x86 なので、A64FX の計算ノードのジョブで入れる (`envs/wis_ody/install_job.sh` が例)。
  LLVM 18 を OS の GCC 8.3.1 で作り、その clang で AdaptiveCpp と numpy・SciPy の C / C++ を作る (GCC 12 などを作ると、その新しい
  libstdc++ を富士通の FCC が読めず、C++ のプログラムをリンクできない)。PARCEL のホスト側は FCC、MPI は `mpiFCC`、LAPACK は富士通の
  `-lfjlapack` を使う。OpenBLAS (numpy・SciPy 用) は Fortran なし、SciPy は 1.13 (GCC 9.1 未満のため) に合わせて numpy を 2.2 に固定している。
  並列数は `site_env.sh` の `PARCEL_STACK_JOBS` (12。32 以上では LLVM のビルドでメモリが足りない)。LLVM のビルドに約 3 時間かかる。
  利用者のプログラムを富士通の frt / fcc で組むときは `-Kopenmp` を付ける (`after_activate.sh` が `FFLAGS` / `CFLAGS` に設定する)。
- 上の 7 環境以外の Linux の計算機では、雛形 `envs/generic/` を写して書き換える (例: `cp -r spack/stack/envs/generic spack/stack/envs/mysite`)。
  `spack.yaml` の先頭のコメントに書き換える箇所と手順がある: コンパイラ (GCC) の版と場所、MPI を使う場合はその版と場所、
  NVIDIA の GPU を使う場合は CUDA の版と場所と `cuda_arch`、x86_64 以外の CPU では LLVM の `targets`。
  モジュールで MPI などを読む計算機では `site_env.sh` に `module load` を書く。
  RHEL 系の OS の Open MPI (`dnf install openmpi-devel`) は `module load mpi/openmpi-x86_64` で読み、ヘッダが `/usr/include/openmpi-x86_64` に
  あって `/usr/lib64/openmpi` の下に無い。そのまま `prefix` にすると MPI ありの PARCEL のインストールが
  `Query of package 'openmpi' for 'headers' failed` で止まるので、`bin`、`lib`、`share`、`include` を 1 つのディレクトリに並べて
  (シンボリックリンク) それを `prefix` にする (`spack.yaml` の MPI の注)。
  `CONCRETIZE_ONLY=1 bash spack/stack/install_stack.sh mysite <入れ先>` で解決だけを試してから入れる。
  この雛形は同梱の `s86`・`miyc` の定義にならって作ったもので、そのままの形ではどの計算機でもインストールを試していない。
  AMD の GPU は `psb` の定義 (ROCm の外部登録) を下敷きにする。
- macOS は spack を使わず Homebrew で入れる (`brew install adaptivecpp open-mpi openblas gcc numpy scipy`。
  Python のパッケージは `python/envs/mac_brew.sh` を読み込んでから `pip install`。回帰試験は `tests/site_config.py` の `mac`)。
