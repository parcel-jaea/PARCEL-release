[English](CHANGELOG.md) | 日本語

# CHANGELOG

## v1.2 (2026-10-23)

PARCEL_SYCL v1.1 からの変更点。

### 追加

- **固有値ソルバ `parcel::krylov::ks_arnoldi`** (Arnoldi + Krylov-Schur リスタート)。
  絶対値最大側の `nev` 個の固有対を Ritz 残差で収束判定しながら求める。リスタートには実 Schur 形を使うので、
  非対称行列の複素共役対の固有値も扱える。`set_shift(σ)` で shift-invert にすると、内部の pgmres で
  (A − σI) z = v を解き、σ 近傍の固有値を求める。直交化は `set_qr_algo()` で選ぶ (`pgmres` と同じ定数。
  `ks_arnoldi` ではどちらも 2 回適用する): `QR_CGS` はブロック版の古典グラムシュミット 2 回 (CGS2、既定)、`QR_MGS` は
  修正グラムシュミット 2 回 (MGS2)。それ以外の値は `std::invalid_argument`。C / Fortran は `parcel_options` の
  `ks_qr_algo`、Python は `eigs()` の `orthogonalization` (`"cgs"` / `"mgs"`)。1 ランクの局所長が小さい
  CPU の構成では MGS2 が速いことがあり、局所長が大きい CPU と GPU では CGS2 が速い。固有値の精度は同等。
  環境変数 `PARCEL_KS_ORTHO` は読まない (設定されていると、最初の `solver()` で rank 0 が標準エラーに警告を 1 行出す)。
- **固有値ソルバ `parcel::krylov::inverse_power`** (逆べき乗法)。絶対値最小の固有値を求める。
  内部の線形ソルバは pgmres (既定) または pcg。
- **前処理後の行列 A K⁻¹ の固有値**: 固有値ソルバ共通の `set_precon_spectrum(true)`。既定 (false) は A の固有値。
- **固有値ソルバの内部線形ソルバの設定**: `set_linsolve_algo / _tolerance / _max_iteration / _restart /
  _precon_algo / _blockjacobi / _jacobi / _scaling_matrix / _scaling_vector`。
- **収束状態の API** (全ソルバ): `get_converged_reason()` (`CONVERGED_RTOL` / `DIVERGED_ITS` /
  `DIVERGED_NANORINF` / `DIVERGED_BREAKDOWN`)、`is_converged()`、`get_iterations()`、`get_residual()`、
  `get_true_residual()`。固有値ソルバは `get_linsolve_failures()`、`ks_arnoldi` は `get_ks_nconv()`。
  rank 0 が `solver()` の最後に ` converged_reason = ... , its = ...` の 1 行を書く。
- **C / Fortran から全 7 ソルバを呼べる API** (`capi/parcel_sycl_c.h`、`capi/parcel_sycl_mod.f90`、
  `capi/wrapper_parcel_sycl.cpp`): `parcel_solve_linear` / `parcel_solve_eigen` (Fortran 用は `_f`)、
  設定の構造体 `parcel_options` (既定値は `parcel_options_init()` がライブラリから読む)、結果の構造体
  `parcel_result`、不均等な行分割、エラーコード (`PARCEL_ERR_*`)。v1.1 の `run_parcel_sycl_c` / `_f` は
  互換のため残した。
- **`pcagmres` の Q の作り方の選択** `set_explicit_q(CAGMRES_Q_AUTO / CAGMRES_Q_EXPLICIT / CAGMRES_Q_IMPLICIT)`。
  `CAGMRES_Q_IMPLICIT` は Q を陽に作らない CholeskyQR 版 (v1.1 の `cholQR_pcagmres` のアルゴリズム)。
  QR を指定しなければ `QR_CHOLESKYQR`、対称行列・前処理なし・s が偶数ならモーメント法
  `QR_CAGMRES_CHOLESKYQR` も選べる。
- **入力の検査**:
  - `set_crsMat()` が行ブロック分割 (全ランクで [0, n) を隙間・重複なく覆う) と局所 CRS (行ポインタ・列番号の範囲) を
    調べ、崩れていれば全ランクが同じ `std::invalid_argument` を投げる (`-DPARCEL_NO_CRS_CHECK=1` で外せる)。
  - 設定関数・`solver()` の引数の範囲外の値は `std::invalid_argument`。
  - SYCL のキューが in-order でなければ `std::invalid_argument` (`-DPARCEL_REQUIRE_IN_ORDER=0` で外せる)。
  - デバイスメモリの確保に失敗したら `util_sycl::device_bad_alloc` (`std::bad_alloc` の派生) を投げる。
  - `-ffast-math` でのビルドを `#error` で止める (double-double と NaN の検出が壊れるため。
    `-DPARCEL_ALLOW_FAST_MATH` で許可)。
- 実行時の環境変数 `PARCEL_TIMER_SYNC=1`: 内積と係数を値で渡す axpy の時間計測の前後でデバイスを同期する
  (既定は同期せず、内積の時間に直前のカーネル (MGS の axpy など) の時間が入る。時間の内訳を正確に取りたいときに使う)。
- 実行時の環境変数 `PARCEL_MEM_REPORT`: 設定すると PARCEL が USM で確保したデバイスメモリの現在量とピークを数える
  (`util_sycl::device_mem_peak_bytes()` で読める。検証サンプルはソルバの後に `PARCEL device peak` として表示する)。
- 検証サンプル (`tests/verify/`) のメモリの報告 (`[MEM]` と `[MEMG]` の `host_hugetlb_mib`) に、大きいページ (HugeTLB) の量 (`/proc/self/status` の
  HugetlbPages、報告した時点の値) を加えた。A64FX (富士通のコンパイラ) はヒープを大きいページに置くので、ピーク常駐メモリ (VmHWM) に数えられない。
  ホストのメモリは VmHWM + HugetlbPages で見る。大きいページを使わない環境では 0。
- **ビルドするソルバを絞るスイッチ** `PARCEL_ENABLE_ALL_SOLVERS` / `PARCEL_ENABLE_<ソルバ>` (既定は全部)。
- `PARCEL_FP_CONTRACT_OFF`: `#pragma clang fp contract(off)` を有効にする (`-ffp-contract=fast-honor-pragmas` と
  併用したときだけ効く。ビルドし直しても結果を一致させたい場合は `-ffp-contract=off` を推奨)。
- **MPI なしのビルド** `-DPARCEL_MPI=0`: `mpi.h` を読まず、MPI のコンパイララッパなしで 1 プロセス版を作れる。
  `parcel::comm comm(&q)` で作り、`set_crsMat(&comm, nstart, nend, ...)` などは MPI ありと同じ形で使える。
  C / Fortran の API も同じ関数で、コミュニケータの引数は使われない。結果は MPI ありの 1 プロセスとビット単位で一致する。
  使い方は README の「MPI なしのビルド」。
- 回帰試験に MPI なしの構成 (`<plat>_nompi`、`-DPARCEL_MPI=0`、1 プロセスを直接起動) を追加。対象は `tests/site_config.py` の
  `REGRESSION_NOMPI`。MPI あり np=1 との突き合わせに `tests/regression/compare_nompi.py` を追加。
- 回帰試験に C / Fortran の API の試験 (`examples/sample_c.c` / `sample_f.f90`) を追加 (`lang_c` / `lang_f`、1 項目 = 1 ケース)。
  ビルドは `tests/regression/sub_build_lang_<plat>.sh`、対象は `tests/site_config.py` の `REGRESSION_LANG`。
  `tests/regression/summarize.sh` は言語別の件数も出す。
- クイックスタートの C / Fortran / Python 版 (`examples/quick_start_c.c`・`quick_start_f.f90`・`quick_start.py`): `quick_start.cpp` と同じ問題
  (1 次元ラプラシアン n = 1000 を前処理なしの pcg で) を C / Fortran の API と Python のパッケージで解く。MPI なし / ありとも同じソース。
- **ビルド後の確認**: クイックスタート 4 本 (C++・C・Fortran・Python) が厳密解 x[i] = (i + 1)(n − i) / 2 との最大相対誤差を
  各ランクの担当行で求めて集め、しきい値 1e-6 で判定する (`check: max relative error = ... (limit 1.0e-06) PASS`。不合格なら終了コード 1)。
  `examples/run_quick_checks.sh` (新規) はこれらをビルドして実行し、MPI ありなら C++ を 2 プロセスでも実行して、
  `PASS n / FAIL m` を 1 行で出す (MPI なし `PARCEL_MPI=0`、起動コマンド `MPIEXEC`・`MPIEXEC_NP_FLAG` など、Python は `parcel` が
  無ければ SKIP)。ビルドは `examples/build/build_stamp.txt` (設定・コンパイラの場所・ソースの cksum) が今と同じなら飛ばし、違えば
  その項目を表示して作り直す (`--rebuild` で必ず作り直す)。`build_examples.sh` は C / Fortran のクイックスタートもビルドし、
  コンパイルを同時に走らせる (`BUILD_JOBS`、既定は対象の数とコア数の小さい方)。対象 `quick` (クイックスタートの 3 本だけ) と
  記録を出す `--print-stamp` を追加。spack のレシピにインストール後の試験 (`spack test run parcel-sycl
  py-parcel-sycl`) を追加し、`py-parcel-sycl` にバリアント `scipy` (既定 on、SciPy を依存に入れる) を追加した。
  スレッド数の既定 (どちらも同じ): `OMP_NUM_THREADS` が無ければ、計算ノード (ジョブの環境変数 `PBS_JOBID`・`PJM_JOBID`・
  `SLURM_JOB_ID` がある) では使える論理 CPU 数のすべて、ログインノードでは 4 (MPI の実行はプロセス数で割る。`QUICK_NODE=compute|login` で
  強制)。判定と使った値を表示する。富士通の mpiexec では `RC_ONLY=1` が既定 (`run_quick_checks.sh` は実体が `/opt/FJSV…` の
  `mpiexec`、spack は `+fj +mpi`)。Open MPI は配置の指定が無ければ `--bind-to none` を足す。spack の試験は Intel MPI では
  C・Fortran を汎用のラッパ (`mpicc`・`mpif90`) と `I_MPI_CC`・`I_MPI_F90` で spack のコンパイラ (gfortran など) で組む
  (`mpiifx` (ifx) は gfortran で作ったモジュールを読めない)。
- **Python のパッケージ `parcel`** (`python/`): C / Fortran の API とは独立したネイティブ層 (`python/src/parcel_py.cpp`) から
  全 7 ソルバを呼ぶ (`parcel.solve` / `parcel.eigs`、倍精度、MPI なし / あり)。pip (scikit-build-core) または spack の
  `py-parcel-sycl` で入れる。説明は `python/README.ja.md`。
- **spack のレシピとゼロから入れる仕組み** (`spack/`): `parcel-sycl` (ヘッダ・C / Fortran の API) と `py-parcel-sycl`、
  spack の取得から LLVM・AdaptiveCpp・OpenBLAS・Python・PARCEL までを入れる `spack/stack/install_stack.sh` と確認の
  `check_stack.sh`、7 環境の定義 (`spack/stack/envs/`。Wisteria-Odyssey は富士通のコンパイラ・MPI・LAPACK を使う `+fj`)。それ以外の Linux の計算機向けに書き換えて使う雛形 `spack/stack/envs/generic/` (Ubuntu 24.04 と Rocky Linux 9 (いずれも WSL2) で確認)。V100 (`cuda_arch=70`) では `-Xcuda-ptxas -maxrregcount=255` を自動で付ける。
  説明は `spack/README.ja.md`。
- 回帰試験に Python の API の試験 (`tests/regression/sample_py.py`。`sample_c.c` と同じ問題・項目・判定) を追加 (`lang_py`)。
  ビルドは `sub_build_lang_<plat>.sh` (ネイティブ層を `bin/<plat>/libparcel_py.so` に作る)、対象は `tests/site_config.py` の
  `REGRESSION_LANG_PY`。C / Fortran / Python の試験は、パッケージのソースを組む (既定) か、spack などでインストールした
  PARCEL を使うかを `parcel_from` で選べる。
- 回帰試験・性能試験を、ゼロから入れた spack の環境で流す設定の例 (`tests/site_config.py` の `s86spack_cpu` / `s86spack_gpu`)。
  回帰試験は Miyabi-C / Miyabi-G / PS-A / PS-B / Wisteria-Aquarius / Wisteria-Odyssey の例 (`miy_cspack` / `miy_gspack` / `ps_aspack` /
  `ps_bspack` / `wis_aquspack` / `wis_odyspack`) もある。Odyssey は富士通のコンパイラ・MPI で組み、富士通の `mpiexec` が
  ジョブのログに書くランクの標準出力を `-std` でファイルに取って判定に使う (`tests/site_config.py` の `rank_out`)。
  どの例も spack に入れた PARCEL を検証サンプル (C++) を含めて試す (`parcel_from="installed"` で検証サンプルと
  ddcheck をインストールしたヘッダで組み、C / Fortran / Python は `parcel-sycl` / `py-parcel-sycl` を使う)。
- 検証サンプルの既定値モード (`PARCEL_LIB_DEFAULTS=1`): ソルバのパラメータの設定関数をサンプルの値で呼ばず、ライブラリの既定で解く
  (環境変数で明示した値はそのまま使う)。実際に使った値を `[LIB_DEFAULTS]` の行に出す。性能試験が使う。
- 回帰試験の MPI あり / なしと言語 (検証サンプル (C++) / C / Fortran / Python) を、生成 (`gen_regression.py --plat --mpi --lang`)・
  ビルド (`sub_build_<plat>.sh`、`sub_build_lang_<plat>.sh [c] [fortran] [python]`)・投入 (`feed_regression.sh` の `--mpi` / `--lang`、
  種別 `cpp` / `lang_c` / `lang_f` / `lang_py`)・集計 (`summarize.sh --mpi --lang`) のどの段階でも独立に選べるようにした。
  選んで生成したときは選んだ組の生成物だけを作り直す。
- スケジューラの無い計算機での回帰試験 (`tests/site_config.py` の `sched="local"`、`tests/regression/run_local.sh` がビルド・
  実行・集計する)。例として macOS + Homebrew の環境 `mac` を追加。
- 例題 `examples/quick_start.cpp` (最小の pcg) と `examples/example_pcg_ks.cpp` (pcg + ks_arnoldi)。
- 回帰試験 (`tests/regression/`) と性能試験 (`tests/perf/`) の生成器、検証サンプル
  (`tests/verify/parcel_krylov_sample_verify.cpp`。直接法・密行列の固有値計算による参照解と比べる)、
  double-double の劣化検出 (`tests/regression/ddcheck/`)。
  回帰試験の低精度の個別の判定値 (`calcF16` / `dataF16` の pcg・pbicgstab) は CPU と GPU で共通。
  回帰試験の反復の上限は線形ソルバ 10000、power 5000 (ブロックヤコビ前処理のときは 20000)、inverse_power 200。
  CPU 環境のランクあたりスレッド数は s86 5、A64FX 12 (1 CMG)、Miyabi-C 7、PS-A 7 (Miyabi-C と PS-A は 1 NUMA に複数ランク)。
  CPU のブロックヤコビ前処理の分割数はランクあたりスレッド数で決まる。
  性能試験は NX=512 を収束まで (相対残差 1e-8、最大 10,000) 解く時間を、装置数 4 / 8 / 16 (CPU はソケット、GPU は 1 台) で
  各 3 回測る (集計は最速の回)。非対称向けのソルバ (pgmres m=60、pcagmres explicit / implicit s = t = 8) は移流拡散
  (c = 0.05)、pcg・pbicgstab と固有値ソルバはポアソン。ソルバのパラメータはライブラリの既定 (inverse_power の内部ソルバだけは
  pcg・ヤコビ前処理を指定する。既定の pgmres は再始動 30 でポアソンの内部の解法が上限に達し、10〜14 倍遅い)。混合精度 (`precon_dataF`) は
  ヤコビ前処理の pcg・pgmres を倍精度と比べる (`tests/perf/gen_perf_prec.py`)。混合精度の実行ファイルは
  `tests/perf/sub_build_prec_<plat>.sh` が `bin/<plat>/perf_prec/` に作り、`precon_dataF` はヤコビ前処理のその場計算なし
  (`-DPARCEL_JACOBI_ONTHEFLY=0`) にする (既定のその場計算では前処理のたびに A の倍精度の値を読み、単精度にする利点が出にくい)。
  比べる相手の倍精度と回帰試験の `precon_dataF` は既定のまま。
  ks_arnoldi の直交化は既定の CGS2 で測るが、PS-A (`ps_a`) だけは MGS2 (検証サンプルの `PARCEL_KS_QR=mgs` が
  `set_qr_algo(QR_MGS)` を呼ぶ。`tests/site_config.py` の `PERF` の `job_args`) で測る。1 スレッドあたりの局所長が小さく大きなキャッシュを持つ CPU では MGS2 が速いことがある。
- `ks_arnoldi` の直交化の既定 (`set_qr_algo()` を呼ばないとき) をビルド時に選ぶマクロ `PARCEL_KS_ORTHO_DEFAULT`
  (`cgs2` / `mgs`、未定義なら CGS2)。spack のレシピにバリアント `ks_ortho` (既定 `cgs2`) を加え、PS-A の定義 (`spack/stack/envs/psa`) と
  試験の `ps_a` / `ps_aspack` は `mgs` にした (PS-A は MGS2 が既定)。マクロを定義しないビルドは前処理後のソースが変わらない。
- 試験の環境設定の例 (`tests/site_config.py`): リンクに LLVM の `libomp.so` の場所の RPATH を付ける (`LINKOPT`。AdaptiveCpp は
  `acpp` でも MPI のラッパ経由でも付けないので、ジョブの `LD_LIBRARY_PATH` が無いと OS の libomp を読む)。PBS の配置
  (`place`) を指定できるようにし、PS-B (1 vnode = 1 APU) は 1 ノードに収まるジョブを `place=pack`、収まらないジョブを
  `place=exclhost` にした (GPU の割り当てがノード内の順番で GPU を選ぶので、散らばると他のジョブの GPU を使う)。
  回帰試験で np が 1 ノードの装置数 (`node_chunks`) を超えるときは、起動コマンドの 1 ノードのランク数を `node_chunks` にして
  ceil(np / node_chunks) ノードに詰める (ノードをまたぐと縮約の順序が変わり、低精度の構成では反復数が変わる)。
- **版の情報**: `Include/parcel_version.hpp` (`parcel_sycl.hpp` が読む) に `PARCEL_VERSION_MAJOR` / `_MINOR` / `_PATCH`、
  `PARCEL_VERSION_STRING` (`"1.2.0"`)、`PARCEL_VERSION_NUMBER`、`PARCEL_VERSION_GE(a, b, c)` と、`parcel::version_string()`・
  `parcel::get_version()`。C / Fortran の API に `parcel_get_version()`・`parcel_version_string()`、Python に `parcel.library_version()`。
  Python のパッケージの版を 0.1.0 から本体と同じ 1.2.0 にした。版を書き写しているファイル (Python のパッケージ・spack のレシピ・
  `CITATION.cff`) は `tests/check_version.py` で照合し、回帰試験は実行した版を生成器がヘッダから与える値と比べる
  (検証サンプルの `[VERSION]` の行、C / Fortran / Python の試験の `version` の項目)。
- 全ソースファイルの先頭に `SPDX-License-Identifier: LGPL-3.0-only` と著作権の行。`CITATION.cff` (引用の情報) と
  `SUPPORT.md` (問い合わせの案内)。
- 配布 tarball を作る `tools/make_tarball.py` (同じ中身から同じ tarball・sha256。名前は `parcel_sycl-1.2.0.tar.gz`)。
  spack のレシピに配布 tarball の `url` (GitHub の Release `https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz`) と、
  `--pin` で書く sha256 (tarball に入れるレシピは sha256 なしの形にする)。spack のレシピ・環境の定義の版は `1.2.0`
  (`version("1.2.0")`・`parcel-sycl@1.2.0`。`tests/check_version.py` が環境の定義と README の `@` の版も照合する)。
- 問い合わせ先は公開用リポジトリの GitHub の Issues (`SUPPORT.md`、`CITATION.cff` の contact)。`CITATION.cff` に公開リポジトリ
  (`https://github.com/parcel-jaea/PARCEL-release`) と引用してほしい論文 (Idomura et al., ScalA 17, 2017)。
- **マニュアルの HTML を同梱** (`doc/`): 利用者ガイドと API リファレンスの日本語版・英語版 (入口 `doc/index.html`、英語は
  `doc/index_en.html`)。ウェブブラウザでそのまま開け、外部のサイトから読み込むものは無い (数式は MathML、検索はページ内の索引)。
  開発者ガイドは含めない。
- **説明の md を英語版と日本語版の 2 本に**: `README.md`・`CHANGELOG.md`・`SUPPORT.md`・`spack/README.md`・`python/README.md`・
  `tests/regression/README.md`・`tests/perf/README.md` を英語版とし、日本語版を同じ場所の `*.ja.md` (例 `README.ja.md`) に置いた。
  各ファイルの先頭に相互のリンク。`CITATION.cff`・`COPYING*` は 1 本のまま。

### 変更 (動作・API)

- NVIDIA の GPU (`+cuda` / `cuda:sm_*`。V100・A100・H100) の推奨から梯子分割 (`-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1`) を外した
  (V100 の `-Xcuda-ptxas -maxrregcount=255` は残す)。V100 では梯子分割が平坦なループより遅く、A100・H100 (GH200) では差が無かったため。
  spack のレシピの `kernel=auto`、Python の `PARCEL_KERNEL_FLAGS` の既定、`tests/site_config.py` の s86 GPU・Miyabi-G・Aquarius
  (`ladder=False`)、README のビルド例を変えた。AMD の GPU (MI300A) は梯子分割のまま (速くなる)。CPU は変わらない。

- macOS + Homebrew の推奨のカーネル最適化を、他の CPU と同じ縮約だけのブロック化 (`-D_BLOCKING_LOOP_DOT_=1`) にした
  (`python/envs/mac_brew.sh`。それまでは A64FX 向けの全カーネルのブロック化 `-D_BLOCKING_LOOP_=1`)。Apple M4 での 1 反復の差は ±4 % 以内で、
  Jacobi 前処理の値を保持しない分メモリが少ない。回帰試験の `mac` はもともとこの設定。Homebrew の AdaptiveCpp では
  `--acpp-targets=omp` と `omp.library-only` は同じコンパイルになる。

- 検証サンプル (`tests/verify/parcel_default.hpp`) と C / Fortran のラッパ (`capi/parcel_capi_default.hpp`) の最大反復回数のマクロ `MAX_ITERATION` の
  既定を 10 から 10000 (ライブラリの `set_max_iteration()` の既定と同じ) に変えた。検証サンプルの `PARCEL_MAX_ITER` の既定と旧 API
  `run_parcel_sycl_c` / `_f` の反復上限が変わる。新 API (`parcel_solve_linear` など) とライブラリ本体は変わらない。

- **既定値** (何も指定しなくても解けるように):

  | 設定 | v1.1 | v1.2 |
  |---|---|---|
  | `set_tolerance()` | 1.0e-4 | 1.0e-8 |
  | `set_max_iteration()` | 100 | 10000 |
  | `pgmres` の `set_restart()` | 4 | 30 |
  | `pcagmres` の `set_restart(s, t)` | (1, 1) | (8, 4) |
  | `pcagmres` の `set_qr_algo()` | `QR_CGS` | `QR_MGS` |

  `pcagmres` の QR は (s, t) = (8, 4) の CGS で最初の周期に残差が大きく増えることがあるので MGS にした
  (`CAGMRES_Q_IMPLICIT` で QR を指定しなければ `QR_CHOLESKYQR` なのは変わらない)。C API の `cagmres_qr_algo` の既定も
  `PARCEL_QR_MGS` で、`CAGMRES_Q_IMPLICIT` のときはこの値を「未指定」とみなす (`PARCEL_QR_CGS` の明示は `PARCEL_ERR_ARG`)。
  v1.2 で加えた `ks_arnoldi` の最大リスタートは 200、固有値ソルバの内部線形ソルバの前処理はヤコビ (反復 1 回)。
  v1.1 と同じ動作にするには上の設定関数で旧値を明示する。`pgmres` は (リスタート周期 + 1) 本の作業ベクトルを
  持つので、既定のメモリ使用量が増える。
- **整数型**: 大域の量 (全体の行数、`nstart` / `nend`、`set_crsMat()` に渡す行ポインタと列番号) の型
  `PARCEL_INT` の既定を `int32_t` から `int64_t` にした (`-DPARCEL_INT32` で 32 ビット。v1.1 の `-DPARCEL_INT` は
  互換のため受け付けて 64 ビットにする)。デバイス上の CRS の添字は新しい型 `PARCEL_LINT` (既定 `int32_t`、
  `-DPARCEL_LOCAL_INT64` で 64 ビット) にし、SpMV のメモリ転送量を 32 ビットのときと同じに保つ。
  1 ランクの非零数 (または局所行数 + 袖の数) が `PARCEL_LINT` に収まらなければ `set_crsMat()` が全ランクで
  `std::overflow_error` を投げる。
- **C / Fortran の API の整数は 64 ビット** (C: `parcel_int` = `int64_t`、Fortran: `integer(parcel_int_kind)` =
  `integer(c_int64_t)`)。旧 API `run_parcel_sycl_c` / `_f` の `n` / `nnz` / `crsCol` / `crsRow_ptr` も同じ
  (**ABI の変更**。ラッパ・Fortran モジュール・利用者のプログラムをそろえて再ビルドすること)。
- **精度のテンプレート引数を省略できる**: `parcel::matrix<>` / `parcel::vector<>` と各ソルバの `solver()` の
  データ精度に既定 `parcel_double` を付けた (`parcel::krylov::pcg<> s; s.solver(A, x, b);` と書ける)。
- **`pbicgstab` のテンプレート引数**: 使っていなかった先頭の `Tdata` を削除し、他のソルバと同じ
  `<Tspmv, Tprecon, Taxpy, Tdot>` にした。v1.1 で 5 つ指定していた場合は先頭を外す
  (v1.1 では 4 つ指定すると 1 つずつずれて、SpMV・前処理・内積が意図と違う精度で計算されていた)。
- **`cholQR_pcagmres` クラスを廃止した。** `pcagmres` に `set_explicit_q(parcel::krylov::CAGMRES_Q_IMPLICIT)` を
  指定する (QR は既定で `QR_CHOLESKYQR`)。`PARCEL_ENABLE_CHOLQR_PCAGMRES` を指定すると `#error` で止まる。
- **`pgmres::solver()` の末尾に引数** `parcel::precon_matrix<...> *external_precon = nullptr` を追加
  (既定値があるので既存の呼び出しはそのまま)。
- **`pgmres` / `pcagmres` は反復上限をリスタート周期の倍数に切り捨てない。** 上限ちょうどまで反復する
  (最後の周期を短くする)。上限が周期の倍数のとき・上限の前に収束するときは結果は変わらない。
- **`power` / `inverse_power` は反復中に NaN / Inf が出たらその反復で止まり** `DIVERGED_NANORINF` を返す
  (最後の有限な固有値と正規化済みのベクトルを返す)。
- **`power` の初期ベクトル**を全要素 1 から、大域行番号から作る sin/cos の組合せに変えた (MPI 分割数によらず同じ)。
  全要素 1 は 3 次元ポアソン (NX が偶数) の最大固有値の固有ベクトルと直交し、その固有値に到達できなかった。
- **`ks_arnoldi` の固有ベクトルの出力** `EigenVec` は n × nev 以上の長さなら受け付け、min(k_keep, 長さ / n) 本だけ書く
  (上位 nev 本だけ要るなら n × k_keep でなく n × nev を確保すればよい)。
- ブロックヤコビのブロック数とスケーリングの既定をデバイスと精度から自動で決める
  (`set_blockjacobi()` / `set_scaling_*()` を呼ばない場合)。
- **既定をメモリの少ない設定にした** (アルゴリズムは同じ。メモリを使って速くする設定は README の「メモリと速さ」):
  - ヤコビ前処理 (`PRECON_JACOBI`) は (D⁻¹)(L+U) の値を保存せず、前処理のたびに A の値と D⁻¹ から計算する
    (`PARCEL_JACOBI_ONTHEFLY`、既定 1)。A の非零 1 個あたり前処理行列の型 1 個ぶんメモリが減る代わりに、GPU では
    前処理が遅くなる (CPU ではほぼ変わらない)。値は構築時と同じ式で作るので、`-ffp-contract=off` なら結果はビット単位で
    同じ。行列スケーリングを使うとき (半精度の前処理では既定で有効)、係数行列・前処理行列・前処理の計算型のいずれかが
    double-double のときは従来どおり値を保存する。値を保存する組み合わせの前処理は、`-DPARCEL_JACOBI_ONTHEFLY=0` の
    ビルドと同じカーネル (保存した値を読むだけ) を使うので遅くならない (どちらのカーネルを使うかは前処理を呼ぶ前に 1 回だけ決める)。
    `-DPARCEL_JACOBI_ONTHEFLY=0` で従来の動作。
    A64FX 向けのビルド (`_BLOCKING_LOOP_` または `_BLOCKING_LOOP_SPMV_` が 1) では既定を 0 (値を保存) にした。A64FX 1 ノード
    (NX=256、pcg + ヤコビ前処理) で値を保存すると前処理が 0.77 倍、収束までの総時間が 0.81 倍になる (その場の計算が隠れない)。
    メモリは A の非零 1 個あたり前処理行列の型 1 個 (倍精度で 8 バイト) 増える。`-DPARCEL_JACOBI_ONTHEFLY=1` を明示すれば省メモリ側。
  - `pcg` が解の更新をまとめる反復数 (pstep) の既定を 4 から 1 にした (x を毎反復更新し、p の履歴を 1 本だけ持つ)。
    ベクトル 3 本分メモリが減る代わりに、`pcg` が数 % 遅くなる。解の足し合わせ方が変わるので丸めの桁で結果が変わる
    (反復数は同じ)。実行時の `PARCEL_PSTEP=4`、またはビルド時の `-DPARCEL_PCG_NO_XHIST=0` で従来の動作。
- ソルバの実行時間の出力を、メインループとその内訳 (dot / axpy / SpMV / 前処理) に分けた。

### 削除

- **`parcel::krylov::pcaarnoldi`** (通信削減型 Arnoldi 法) とマクロ `CAARNOLDI_RESTART_S` / `_T` / `CAARNOLDI_QR_ALGO`。
  リスタートと収束判定を持たず、絶対値最大側の数個以外の Ritz 値の精度が保証されなかった。同じ用途は
  `ks_arnoldi` が置き換える。

### 修正

- `pcagmres` の解の更新 (`update_xvector`) が、作業用の BLAS の通信子に行列の通信子ではなく `MPI_COMM_WORLD` を使っていた。
  行列の通信子 (`Mat.comm()->mpi_comm()`) を渡すようにした。この中の演算 (gemv・axpy) は通信しないので、結果は変わらない。
- 富士通コンパイラ (A64FX の `omp.library-only` のホストコンパイラ。clang モードは LLVM 7 ベース) で、double-double の演算のための
  `#pragma clang fp contract(off)` が `Include/ddlib.hpp` の範囲で戻らず、翻訳単位の残り (SpMV・axpy・内積・前処理などの全カーネル) で
  FMA 命令が使われていなかった (`-ffp-contract=fast` を付けても同じ)。この版は `float_control` の push / pop を知らず、pop で戻らない。
  FMA の禁止を double-double の関数の本体の中 (複合文の先頭の pragma。clang の全版でその複合文の中だけに効く) に限った。
  double-double の関数の生成コードは変わらない (富士通コンパイラと clang 18 でアセンブリが一致)。A64FX では推奨のフラグ (`-ffp-contract=fast`、下の「性能・メモリ」) で
  他のカーネルが FMA を使うようになり、結果の最後の桁が変わる。clang 11 以降 (AdaptiveCpp の x86・GPU) は pop で戻っていたので変わらない。
- Fortran のモジュール (`capi/parcel_sycl_mod.f90`) を富士通 Fortran (frt) でもコンパイルできるようにした。`parcel_solve_linear` / `parcel_solve_eigen` は
  モジュールの Fortran の手続きになった (名前・引数・`res` の省略可は同じ。BIND(C) の手続きの OPTIONAL な引数は Fortran 2018 の機能で, frt が受け付けないため)。
- `util_sycl::vector` の `resize` / 解放が、他の配列の領域を指す所有しない view (`shallowcp`) の領域まで解放していた。
- デバイスメモリを確保時に 0 で初期化するようにした (未初期化の値が NaN / Inf を含むと 0 × NaN で結果が壊れていた。
  単精度・半精度・double-double の構成で顕在化)。
- 袖通信 (`SpMVcomm`) の `MPI_Wait` の条件が投入条件と入れ違っていた (送受信数が非対称なランク対で不正な待ちになる)。
- 行ブロックがランク番号の順に並んでいない分割で SpMV が誤った結果を返していた。
- 点ヤコビ前処理が袖の長さぶん範囲外を参照していた。
- `pbicgstab` のテンプレート引数のずれ (上記)。係数ベクトルの型を `Tdot` にそろえ、double-double でビルドできるようにした。
- double-double の gemv が係数ベクトルの下位部を落としていた (pcg の解の更新と、QR に CGS を指定した pgmres の直交化)。
- double-double × ヤコビ前処理 (CRS) で、行列スケーリングが無効のとき結果が壊れていた。
- A64FX 向けの `tgemm` の範囲外書き込み。
- `ks_arnoldi`: 複素固有値のリスタートで部分空間の縮約が直交でなかった (実 Schur 形に変更)。低精度の基底で偽の
  不変部分空間を「収束」と報告しうる問題 (真の残差で確かめ、確かめられなければ乱数方向で基底を広げる)。
  混合精度でビルドできない問題。
- v1.1 の `LAPACKE_dgeev` の `ldvl` の誤り (`pcaarnoldi` とともに削除)。
- 初期値 `x0` が厳密解 (初期残差 b − A x0 = 0) のとき、`pcg` が `DIVERGED_NANORINF`、`pbicgstab` が `DIVERGED_BREAKDOWN` を
  返していた。初期の相対残差 ‖b − A x0‖/‖b‖ が `tol` 未満なら、全線形ソルバが反復せずに `CONVERGED_RTOL`・反復回数 0 を返し、
  `x0` を変えない (`get_residual()` / `get_true_residual()` は初期の相対残差)。初期残差が 0 でなく `tol` 未満のとき、
  `pcg` / `pbicgstab` は 1 回反復してから収束していた (反復回数 1 → 0)。
- `pcagmres` の階数落ちの閾値を、基底を低精度で持つ構成 (基底が半精度・単精度) で過大に見積もっていた。正常な列まで
  階数落ちと判定し、各周期で基底を 1 列しか使わずに回っていた (反復数が大きく増える)。閾値を、QR の計算精度で決まる値
  (1e3 u、CholeskyQR は 1e2 √u) と基底の格納精度の揺らぎ 10 u_basis の大きい方にした。基底と QR の計算が同じ精度なら
  閾値は変わらない。なお、半精度の基底では CholeskyQR 系 (`CAGMRES_Q_IMPLICIT`) は使えない (README の「精度」)。

- GPU で `pcagmres` の `CAGMRES_Q_IMPLICIT` + `QR_CAGMRES_CHOLESKYQR` (モーメント法) が s = 4, 8 のとき収束が遅くなる、
  または発散していた。内積に使う行列積 (4 行 × 奇数列の専用経路) が並列の部分和の一部しか結果に入れていなかった。

- 対称な行列積 (CholeskyQR が使う、A64FX 以外の版) が C に beta を掛けずにワークアイテム数倍して足していた
  (既存の呼び出しは C = 0・beta = 0 なので結果は変わらない)。
- DIA 行列のブロックヤコビ前処理の構築で、各ブロックの処理が行列全体を写していた (結果は同じ。GPU で行数の 2 乗の手間)。

- 1 ランクから複数の GPU が見え、行列のキューに既定以外の GPU を選ぶと、通信と内積の作業域が別の GPU に確保され、
  不正なメモリアクセスで止まっていた。作業域を行列 (または演算) のキューで確保するようにした。
- ブロックヤコビ前処理 (CRS) の構築で、特定の CPU 向けのカーネルだけが使う作業域を他の環境でも確保していた (GPU で行数の 256 倍のバイト数)。

### 性能・メモリ

- SpMV が参照しない n に比例する作業配列 (7 点ステンシルで 1 点あたり 1,792 バイト) を確保しないようにした。
  GPU 1 台で扱える問題の大きさが増えた。
- CPU (OpenMP バックエンド) 向けに縮約をブロック化するスイッチ `_BLOCKING_LOOP_DOT_` を加えた
  (barrier の固定費を避ける)。分割数は実行時に OpenMP のスレッド数から決める (環境変数 `PARCEL_NBLOCK` で上書き)。
- GPU 向けに SpMV と前処理の内側ループを梯子状に分割するスイッチ `PARCEL_SPMV_LADDER` / `PARCEL_PRECON_LADDER` を加えた。
- A64FX 向けに gemv を axpy の繰り返しに分解する `PARCEL_GEMV_AS_AXPY` (A64FX のビルドで既定 1)。
- A64FX の推奨のフラグに `-ffp-contract=fast` を加えた (README の「ビルド」、spack のレシピの `+fj`、試験の設定例 `tests/site_config.py` の
  `wis_ody` / `wis_odyspack`)。富士通コンパイラの clang モードは既定で積和を FMA 命令にまとめず、AdaptiveCpp は `-Kfast` の後ろに `-O3` を
  付けるので `-Kfast` も効かず、A64FX のカーネルに FMA が出ていなかった。A64FX 1 ノード (NX=256、pcg + ヤコビ前処理) で 1 反復が 3 %
  (SpMV は 10 %) 速くなる。反復数は同じで、結果の最後の桁が変わる (相対 1e-11 程度)。double-double の関数の中は FMA を禁じたまま。
- A64FX 向けの最適化 (A64FX のビルド `_BLOCKING_LOOP_` だけ。x86・GPU のビルドの生成コードは変わらない。README の「ビルド」):
  - 推奨の実行時の設定に `OMP_WAIT_POLICY=ACTIVE` と `KMP_BLOCKTIME=infinite` を加えた (OpenMP のスレッドを待ち合わせの間も休ませず、
    カーネルの起動と待ちの固定費を減らす。コアを取り合う配置では逆効果。試験の設定例 `tests/site_config.py` の `wis_ody` / `wis_odyspack` に設定)。
  - 係数を値で渡す axpy (`axpy_value` / `axpby_value`) と gemv の axpy 分解版を書き直した (連続した範囲の分割、ポインタと係数のローカル変数への写し、
    富士通コンパイラが元の式に選ぶ融合を `std::fma` で明示、beta = 1 の経路)。`PARCEL_A64FX_AXPY_C` (既定 1。
    `-DPARCEL_A64FX_AXPY_C=0` で書き直す前の形に戻る。結果はどちらでもビット単位で同じ)。
  - `pcg` (pstep = 1) の x と p の更新、r の更新と ‖r‖ の内積をそれぞれ 1 本のカーネルにまとめる `PARCEL_CG_FUSED` (既定 1、`-DPARCEL_CG_FUSED=0` で切れる)。
  - 結果は書き直す前とビット単位で同じ。A64FX (pcg + ヤコビ前処理、200 反復固定の 1 反復) で 1 ノードの NX=64 が 0.59 倍、NX=128 が 0.78 倍、
    NX=256 が 0.99 倍、NX=512 の 8 ノードが 0.84 倍 (A だけのビルドとの比)。
  - ヤコビ前処理の右辺の準備 (D⁻¹b) で、割る値が 1 (ベクトルのスケーリングなし。倍精度・単精度の既定) のときは割り算をしない
    `PARCEL_JACOBI_NODIV_UNIT` (既定 1)。A64FX の SVE の割り算はパイプライン化されず、前処理の約 3 割を占めていた。
  - 前処理行列が単精度 (`precon_dataF` など) のとき、ヤコビ前処理の反復で 2 行を交互に処理して積和の依存の待ちを重ねる
    `PARCEL_JACOBI_ROW2` (既定 1。倍精度の行列・その場計算・`PARCEL_PRECON_LADDER=1` では使わない)。
  - この 2 つも結果はビット単位で同じ (収束までの残差の推移で確認)。A64FX 1 ノード (NX=256、pcg + ヤコビ前処理) で前処理の区間が
    倍精度 0.77 倍・`precon_dataF` 0.70 倍、1 反復が 0.90 倍・0.87 倍。
- V100 (sm_70) のビルドに `-Xcuda-ptxas -maxrregcount=255` を推奨 (README の「ビルド」)。ptxas の既定のレジスタ割り当てでは
  Jacobi 前処理などのカーネルで読み出しが分かれて遅くなる。spill は出ず、結果は変わらない。試験の設定例 (`tests/site_config.py`) の
  V100 の環境に付けた。
- `ks_arnoldi` の直交化をブロック化し、allreduce を束ねた。
- `pcagmres` の直交化の内積 (Q を陽に作る版の R = Qᵀ V と、`CAGMRES_Q_IMPLICIT` のグラム行列) を、
  出力 1 要素に 1 スレッドの gemm (転置) から局所の内積 (dot) の繰り返し + allreduce 1 回にした
  (起動スレッドが出力の要素数しかなく、GPU で遅かった)。Q を陽に作る版の更新 V -= Q R は、1 行に 1 スレッド
  (A64FX は行ブロック) の縦長 gemm にした (Q の各行を 1 回だけ読む)。更新の足し算の順序は変わらない。
- **作業領域の削減** (演算の式と順序は変えていない):
  - ヤコビ前処理 (`PRECON_JACOBI`) の (D⁻¹)(L+U) は行列 A の行ポインタ・列番号を共有し、値だけを A と同じ並び
    (対角の位置は 0) で持つ (列番号と行ポインタの複製をやめた)。反復では行ごとに控えた対角の位置 (1 行 1 バイト) で
    対角を飛ばす。既定では値も持たず、前処理のたびに A の値から計算する (上の「変更」)。前処理は行列 A より長く生きては
    いけない。反復回数 1 (既定) では反復用の作業ベクトル 2 本を確保しない。
  - `pgmres`: 使う区間が重ならない作業ベクトル (残差と解の更新、真の残差) を同じ領域に置き、袖付きベクトルを 4 本減らした。
  - `pcg`: 真の残差を q の領域に置く (1 本減)。既定の pstep = 1 では p の履歴が 1 本で、pstep = 4 よりさらに 3 本少ない
    (上の「変更」)。
  - `pbicgstab`: 生存区間の重ならない作業ベクトルを同じ領域に置き、袖付きベクトルを 11 本から 8 本にした。
  - `pcagmres`: Q を陽に作る版で QR が MGS / CGS のとき、基底ブロック V を Q の中に直接作る (V の s+1 本を確保しない)。
    解の更新の作業ベクトル (`CAGMRES_Q_IMPLICIT` ではブロックの始点のベクトル) は残差の領域を使う。
  - `ks_arnoldi`: リスタートの縮約 V Z を V の中で行い、2 本目の基底 (m+1 列) を真の残差の確認用の 3 列にした
    (double-double を含まず、k_keep+1 <= 32 のとき)。
- **呼び出しごとの固定費の削減** (演算の式と順序は変えていない。`-ffp-contract=off` なら結果はビット単位で同じ。
  積和の融合を許すビルドでは、コンパイラの FMA のまとめ方が変わって丸めや反復数が変わることがある):
  - 内積・max・Gram 行列の gemm の作業域を呼ぶたびに確保せず使い回す。
  - `pcg` / `pbicgstab` / `pgmres` (MGS) / `ks_arnoldi` (MGS) の係数を 1 要素のデバイス配列へ書き戻さず、値のまま axpy に渡す
    (double-double を含む組み合わせは従来どおり)。内積の時間計測の前後の同期を既定で省く (`PARCEL_TIMER_SYNC`)。
  - CPU (`_BLOCKING_LOOP_DOT_` / `_BLOCKING_LOOP_`): MGS の「前の列の axpy」と「次の列の内積」を 1 本のカーネルにまとめる。
    `ks_arnoldi` の CGS2 で局所内積 j+1 本を 1 本のカーネルで取る。gemv と縦長 gemm を行ブロック単位で回す。
    Krylov 基底の列の間隔に 64 バイトの詰め物を入れる (キャッシュセットの衝突を避ける)。
  - A64FX (`_BLOCKING_LOOP_`): SpMV とヤコビ前処理を連続した行の範囲に分け、配列を restrict 付きのローカル変数に写す。
    SpMV は beta = 0 のとき y を読まない (y が NaN / Inf のときと結果が −0 のときの符号だけが変わりうる)。
