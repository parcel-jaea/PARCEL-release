[English](README.md) | 日本語

# PARCEL_SYCL v1.2

PARCEL_SYCL は、SYCL で書かれた並列 Krylov 部分空間法のライブラリ (PARCEL: Parallel Computing Elements) である。
疎行列 (CRS 形式) の連立一次方程式と固有値問題を、MPI による分散並列と SYCL のデバイス
(CPU の OpenMP バックエンド、NVIDIA / AMD の GPU など) で解く。ライブラリ本体はヘッダだけでできている。

v1.2 は v1.1 に次を加えた版である (詳細は [CHANGELOG.ja.md](CHANGELOG.ja.md))。

- 固有値ソルバ `ks_arnoldi` (Arnoldi + Krylov-Schur リスタート) と `inverse_power` (逆べき乗法) の追加。
  v1.1 の `pcaarnoldi` は削除した。
- C / Fortran から全 7 ソルバを呼べる API (`parcel_solve_linear` / `parcel_solve_eigen`)。
- 収束状態を返す API (`get_converged_reason()` など) と、入力の検査 (行ブロック分割・CRS・引数)。
- 行番号などの整数型の既定を 64 ビットにし、デバイス上の CRS の添字だけ 32 ビットにした (`PARCEL_INT` / `PARCEL_LINT`)。
- 設定の既定値を、何も指定しなくても解けるものに変えた (収束判定値 1e-8、反復上限 10000 など)。
- 不具合の修正と、CPU・GPU での性能改善。回帰試験・性能試験の一式。

マニュアル (利用者ガイド・API リファレンス) は `doc/` に HTML で入っている (日本語 [doc/index.html](doc/index.html)、英語 [doc/index_en.html](doc/index_en.html)。
ウェブブラウザでそのまま開ける。外部のサイトには接続しない)。

目次: [ソルバ](#ソルバ) / [精度](#精度) / [必要なもの](#必要なもの) / [ビルド](#ビルド) / [主な既定値](#主な既定値) /
[使い方の最小例](#使い方の最小例) / [注意](#注意) / [メモリと速さ](#メモリと速さ) /
[spack によるインストール](#spack-によるインストール) / [Python](#python) / [試験](#試験) / [ディレクトリ構成](#ディレクトリ構成) /
[版・引用・問い合わせ](#版引用問い合わせ) / [配布 tarball](#配布-tarball) / [ライセンス](#ライセンス)

## ソルバ

| 種類 | クラス (`parcel::krylov::`) | 求めるもの |
|---|---|---|
| 線形 | `pcg` | 前処理付き共役勾配法 (A は対称正定値) |
| 線形 | `pbicgstab` | 前処理付き BiCGStab 法 |
| 線形 | `pgmres` | 前処理付き GMRES 法 (リスタート付き) |
| 線形 | `pcagmres` | 前処理付き通信削減型 GMRES 法 (s ステップ, リスタート付き) |
| 固有値 | `power` | べき乗法。絶対値最大の固有値 1 個 |
| 固有値 | `inverse_power` | 逆べき乗法。絶対値最小の固有値 1 個 (内部で線形方程式を解く) |
| 固有値 | `ks_arnoldi` | Arnoldi + Krylov-Schur。絶対値最大側の `nev` 個、または `set_shift(σ)` (shift-invert) で σ 近傍の `nev` 個 |

- `ks_arnoldi` は複素共役対の固有値も求められる。`power` / `inverse_power` は実数のレイリー商を使うので
  複素固有値には収束しない。
- `pcagmres` は直交基底 Q を陽に作るかを `set_explicit_q()` で選べる
  (`CAGMRES_Q_AUTO` (既定) / `CAGMRES_Q_EXPLICIT` / `CAGMRES_Q_IMPLICIT`)。
  `CAGMRES_Q_IMPLICIT` は Q を作らない CholeskyQR 版で、v1.1 の `cholQR_pcagmres` に当たる。
  QR (`set_qr_algo()`) の既定は `QR_MGS` (`CAGMRES_Q_IMPLICIT` で指定しなければ `QR_CHOLESKYQR`)。
- 固有値ソルバは既定で A の固有値を返す。前処理後の行列 A K⁻¹ の固有値は `set_precon_spectrum(true)` で求める。
- `ks_arnoldi` の直交化は `set_qr_algo()` で選ぶ (`pgmres` と同じ定数。`ks_arnoldi` ではどちらも 2 回適用する)。
  `QR_CGS` はブロック版の古典グラムシュミット 2 回 (CGS2、既定)、`QR_MGS` は修正グラムシュミット 2 回 (MGS2)。
  1 ランクの局所長が小さい CPU の構成では MGS2 が速いことがあり、局所長が大きい CPU と GPU では CGS2 が速い。
  C / Fortran は `parcel_options.ks_qr_algo`、Python は `eigs(..., orthogonalization="cgs" | "mgs")`。
  ビルド時に `-DPARCEL_KS_ORTHO_DEFAULT=mgs` を付けると、`set_qr_algo()` を呼ばないときの既定が MGS2 になる。
  PS-A 向けのビルド (spack の `ks_ortho=mgs`、試験の `ps_a`) は MGS2 が既定。固有ベクトルの出力 `EigenVec` は n × nev 以上の長さを受け付け、
  min(k_keep, 長さ / n) 本だけ書く。

### 前処理

`set_precon_algo()` で選ぶ (C API は `parcel_options.precon_algo`)。

| 値 | 定数 | 内容 |
|---:|---|---|
| 0 | `PRECON_NONE` | なし (既定) |
| 1 | `PRECON_POINTJACOBI` | 点ヤコビ (対角スケーリング) |
| 2 | `PRECON_BLOCKJACOBI` | ブロックヤコビ (ブロックごとの ILU(0))。ブロック数は `set_blockjacobi()`、既定はデバイスから自動 (CPU はランクあたりのスレッド数、GPU は 1 ブロック 2 行) |
| 4 | `PRECON_JACOBI` | ヤコビ反復 (`set_jacobi()` で反復回数、既定 1) |

行列・ベクトルのスケーリングは `set_scaling_matrix()` / `set_scaling_vector()` で指定する
(指定しなければ、ブロックヤコビと低精度の構成で自動的に有効になる)。

## 精度

行列・ベクトルのデータ精度と、演算 (SpMV・前処理・ベクトル演算・内積) の精度を別々に選べる。
型は `parcel_double` (倍精度)、`parcel_float` (単精度)、`parcel_fp16` (半精度)、
`parcel_doubledouble` (double-double、約 32 桁)。

- ソルバのクラステンプレートの引数が演算精度、`solver()` のテンプレート引数がデータ精度である。
  どちらも省略すれば倍精度になる (`parcel::krylov::pcg<> s; s.solver(A, x, b);`)。
  ```cpp
  // SpMV・前処理・ベクトル演算・内積の精度
  parcel::krylov::pcg<parcel_float, parcel_float, parcel_double, parcel_double> s;
  ```
- 検証サンプル・C / Fortran のラッパでは、コンパイル時マクロ `parcel_solver_data_type` /
  `parcel_solver_calc_type` / `parcel_solver_calc_precon_type` / `parcel_solver_data_precon_mat_type` /
  `parcel_solver_data_precon_vec_type` で選ぶ (例: `-Dparcel_solver_calc_type=parcel_doubledouble`)。
- 低精度で到達できる相対残差には限りがある (目安: 単精度で 1e-6、半精度で 1e-3〜1e-2。半精度の前処理は
  1e-6 付近で破綻しうる)。
- 半精度のデータ・演算精度では、`pcagmres` の CholeskyQR 系 (`CAGMRES_Q_IMPLICIT` とモーメント法) を使わない
  (s, t によらず NaN か未収束になる)。Q を陽に作る版 (既定の `CAGMRES_Q_AUTO` で QR が CholeskyQR 以外) は既定の (s, t) で使える。
- 低精度では `pgmres` の残差が判定値のそばで停滞することがあり、リスタート周期を短くすると早く止まる場合がある
  (半精度の演算で m = 30 より m = 15 の方が早く収束した例がある)。
- double-double でも固有値ソルバの固有値は倍精度どまり (ホスト側のスカラが倍精度のため)。

## 必要なもの

- SYCL 2020 の処理系 (動作確認は AdaptiveCpp 24.10。CPU は OpenMP バックエンド、GPU は CUDA / HIP)
- MPI (MPI 並列で使う場合。`-DPARCEL_MPI=0` で MPI なしの 1 プロセス版もビルドできる。下の「MPI なしのビルド」)
- LAPACKE (`lapacke.h` とその実装。例: OpenBLAS)。固有値ソルバが使うので、線形方程式だけを解く場合も必要
- C / Fortran の API を使う場合は C / Fortran のコンパイラ (動作確認は gcc / gfortran)
- Python のパッケージを使う場合は Python 3.8 以上と numpy (下の「Python」)

これらを spack でまとめて入れることもできる (下の「spack によるインストール」)。

## ビルド

ヘッダオンリーなので、ライブラリ自体のビルドは要らない。利用者のプログラムで
`Include/parcel_sycl.hpp` を include し、SYCL コンパイラでコンパイルして LAPACKE をリンクする。
以下は AdaptiveCpp + Open MPI + OpenBLAS の例 (`mpicxx` の中身を `acpp` にする)。

```sh
export OMPI_CXX=acpp        # Open MPI の mpicxx が acpp を呼ぶようにする (MPICH なら MPICH_CXX)

# CPU (OpenMP バックエンド)
mpicxx -O3 --acpp-targets=omp -march=native -D_BLOCKING_LOOP_DOT_=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

# GPU (NVIDIA V100 = sm_70。A100 は sm_80、H100 は sm_90)
#   NVIDIA の GPU には梯子分割 (-DPARCEL_SPMV_LADDER=1) を付けない (下記)。V100 では -Xcuda-ptxas -maxrregcount=255 を推奨する
mpicxx -O3 --acpp-targets=cuda:sm_70 -Xcuda-ptxas -maxrregcount=255 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

# GPU (AMD MI300A。梯子分割を付ける)
mpicxx -O3 --acpp-targets=hip:gfx942 -DPARCEL_SPMV_LADDER=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

mpirun -n 4 ./quick_start
```

CPU (OpenMP バックエンド) の実行ファイルは LLVM の `libomp.so` を使うが、AdaptiveCpp は `acpp` を直接呼んでも
`mpicxx` 経由でも、その場所を実行ファイルに埋め込まない (埋め込むのは AdaptiveCpp 自身のライブラリの場所だけ)。
実行時に LLVM の `lib/` を `LD_LIBRARY_PATH` に入れるか、リンク時に RPATH を付ける。どちらも無いと OS の libomp を
読むか、見つからずに起動できない。

```sh
# llvm_root は AdaptiveCpp を作った LLVM の場所 (spack なら llvm_root=$(spack location -i llvm))
mpicxx -O3 --acpp-targets=omp -march=native -D_BLOCKING_LOOP_DOT_=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas -Wl,-rpath,$llvm_root/lib
```

spack で入れた場合は `$PARCEL_LDFLAGS` (下の「spack によるインストール」) にこの RPATH が入っている。

V100 (sm_70) では `-Xcuda-ptxas -maxrregcount=255` を付けることを推奨する。ptxas の既定のレジスタ割り当てでは、
Jacobi 前処理などのカーネルで行列・ベクトルの読み出しが分かれて直列になり遅くなる。このフラグで読み出しがまとまる
(spill は出ず、計算結果は変わらない)。他の GPU では効果が GPU の世代によるので、必要なら測って決める。

梯子分割 (`-DPARCEL_SPMV_LADDER=1`。前処理の `PARCEL_PRECON_LADDER` も追従する) は SpMV と前処理の内側ループを 4-2-1 に分ける。
AMD の MI300A では速くなる (pcg + ヤコビ前処理の 1 反復 0.66 倍) ので付ける。NVIDIA の GPU では V100 で平坦なループより遅く
(同じ比較で 1.18 倍)、A100・H100 (GH200) では差が無かったので付けない。spack のレシピ (`kernel=auto`) と Python の
`PARCEL_KERNEL_FLAGS` の既定もこの通り。

A64FX (富士通コンパイラをホストコンパイラにする `--acpp-targets=omp.library-only`) では
`-D_BLOCKING_LOOP_=1 -DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=0 -ffp-contract=fast` を付ける (全カーネルのブロック化と SpMV だけの梯子分割、
積和の FMA 命令へのまとめ)。富士通コンパイラの clang モードは既定で積和をまとめず、AdaptiveCpp は `ACPP_OMP_CXX_FLAGS` の `-Kfast` の
後ろに `-O3` を付けるので `-Kfast` も効かない。`-ffp-contract=fast` はコマンドラインに置けば富士通コンパイラまで届く。
A64FX 1 ノード (NX=256、pcg + ヤコビ前処理) で 1 反復が 3 % (SpMV は 10 %) 速くなる。反復数は同じで、結果の最後の桁が変わる
(相対 1e-11 程度)。double-double の関数の中は `Include/ddlib.hpp` が FMA を禁じたまま。
このビルドではヤコビ前処理の値を保存するのが既定になる (`PARCEL_JACOBI_ONTHEFLY` の既定 0、その場計算なし。下の「メモリと速さ」)。
A64FX の推奨はまとめると次のとおり (C2・D・E・F はこのビルドで自動的に有効になる。x86・GPU のビルドの生成コードは変わらない)。

| | 内容 | 指定 |
|---|---|---|
| A | 積和を FMA 命令にまとめる | ビルド時 `-ffp-contract=fast` |
| B | OpenMP のスレッドを待ち合わせの間も休ませない (カーネルの起動と待ちの固定費が減る) | 実行時 `OMP_WAIT_POLICY=ACTIVE` と `KMP_BLOCKTIME=infinite` |
| C2 | 係数を値で渡す axpy と gemv (axpy 分解版) を、連続した範囲の分割・ローカル変数への写し・明示の `std::fma` で書き直したもの | 既定 (`PARCEL_A64FX_AXPY_C`、`-DPARCEL_A64FX_AXPY_C=0` で切れる) |
| D | `pcg` (pstep = 1) の x と p の更新、r の更新と ‖r‖ の内積をそれぞれ 1 本のカーネルにまとめる | 既定 (`PARCEL_CG_FUSED`、`-DPARCEL_CG_FUSED=0` で切れる) |
| E | ヤコビ前処理の右辺の準備 (D⁻¹b) で、ベクトルのスケーリングが無い (割る値が 1) ときは割り算をしない。A64FX の割り算は前処理の約 3 割を占めていた | 既定 (`PARCEL_JACOBI_NODIV_UNIT`、`-DPARCEL_JACOBI_NODIV_UNIT=0` で切れる) |
| F | 前処理行列が単精度 (`precon_dataF` など) のとき、ヤコビ前処理の反復で 2 行を交互に処理して積和の待ちを重ねる (倍精度の行列では使わない) | 既定 (`PARCEL_JACOBI_ROW2`、`-DPARCEL_JACOBI_ROW2=0` で切れる) |
| | ヤコビ前処理の値を保存する (その場計算なし) | 既定 (`PARCEL_JACOBI_ONTHEFLY` 0) |

C2・D・E・F は各要素の式と足す順を変えないので、結果は書き直す前とビット単位で同じ (A64FX、C2・D は 200 反復、E・F は収束まで (306 反復) の残差の推移で確認)。
E・F の効果 (A64FX 1 ノード、NX=256、pcg + ヤコビ前処理、1 反復): 前処理の区間が倍精度で 0.77 倍、`precon_dataF` で 0.70 倍、
1 反復が 0.90 倍・0.87 倍。`precon_dataF` の前処理は倍精度の 1.06 倍になる (E・F の前は 1.17 倍)。
B はスレッドが回り続けるので、コアを取り合う配置 (1 コアに複数のスレッドやほかのプロセス) では逆効果になる。推奨の配置
(1 ランク = 1 CMG・12 スレッド) では問題ない。試験の設定例 (`tests/site_config.py` の `wis_ody` / `wis_odyspack`) は B を設定する。
効果 (A64FX、pcg + ヤコビ前処理、200 反復固定の 1 反復の時間、A だけのビルドとの比): 1 ノード (4 ランク × 12 スレッド) で
NX=64 が 0.59 倍、NX=128 が 0.78 倍、NX=256 が 0.99 倍 (行数が少なく固定費が効く問題ほど効く)。NX=512 では 4 ノードで 0.96 倍、
8 ノードで 0.84 倍で、8 ノードでは PETSc (同じノード数のフラット MPI) より 4 % 速い。

C / Fortran の API はラッパ `capi/wrapper_parcel_sycl.cpp` を SYCL コンパイラでコンパイルし、
利用者のプログラムとリンクする (リンクは SYCL コンパイラで行う)。

```sh
mpicxx -O3 --acpp-targets=omp -I/usr/include/openblas -c capi/wrapper_parcel_sycl.cpp -o wrapper_parcel_sycl.o

# C
mpicc -O2 -fPIE -Icapi -c examples/sample_c.c -o sample_c.o
mpicxx --acpp-targets=omp -o sample_c sample_c.o wrapper_parcel_sycl.o -lopenblas -lm

# Fortran (モジュール capi/parcel_sycl_mod.f90 を先にコンパイルする)
mpif90 -O2 -fPIE -c capi/parcel_sycl_mod.f90
mpif90 -O2 -fPIE -c examples/sample_f.f90
mpicxx --acpp-targets=omp -o sample_f sample_f.o parcel_sycl_mod.o wrapper_parcel_sycl.o \
  $(mpif90 --showme:link) -lgfortran -lopenblas
```

AdaptiveCpp は PIE でリンクするので、gcc / gfortran のオブジェクトは `-fPIE` で作る。
ラッパの内部精度は、ラッパをコンパイルするときの `parcel_solver_*_type` で決まる
(既定値は `capi/parcel_capi_default.hpp`)。examples の C++・C・Fortran のサンプルは `bash examples/build_examples.sh` でまとめて
ビルドできる (コンパイラ・ターゲット・LAPACK は環境変数で変える。スクリプトの先頭を参照)。

### ビルド後の確認

`bash examples/run_quick_checks.sh` で、クイックスタート (C++・C・Fortran・Python、MPI ありなら C++ を 2 プロセスでも) を
ビルドして実行し、1 次元ラプラシアン (n=1000) の厳密解との最大相対誤差が 1e-6 以下かを判定する。最後に
`PASS n / FAIL m` を 1 行で出し、全部合格なら終了コード 0。ビルド (`build_examples.sh quick`。SYCL のコンパイルを同時に
走らせる) は初回だけで、`examples/build/` に前回のビルドの設定とソースの記録があり今と同じなら飛ばして実行だけをする
(実行は 1 分かからない)。設定 (`SYCL_TARGET`・`CXXFLAGS`・`PARCEL_MPI`・コンパイラなど) かソース (ヘッダ・API・サンプル) が
変わっていれば、その項目を表示して作り直す。`--rebuild` で必ず作り直す。
環境変数は `build_examples.sh` と同じ (MPI なしは `PARCEL_MPI=0`、同時に走らせる数は `BUILD_JOBS`)。起動コマンドは
`MPIEXEC` (既定 `mpirun`)・`MPIEXEC_NP_FLAG` (`-n`) で変える。Python は `parcel` を import できなければ SKIP。詳しくはスクリプトの先頭。

**スレッド数と実行場所**: `OMP_NUM_THREADS` が未設定なら、計算ノード (ジョブの中) では使える論理 CPU 数をすべて、
ログインノードでは 4 (使える数が 4 未満ならその数) を使う。MPI の実行では合計がそれになるようプロセス数で割る
(最低 1。ログインノードで 2 プロセスなら 1 プロセス 2 スレッド)。使える論理 CPU 数は、CPU の割り当て (`nproc`)・
cgroup の CPU の上限・ジョブのこのノードの割り当て (PBS の `NCPUS`、Slurm の `SLURM_CPUS_ON_NODE`) の最も小さいもの。
計算ノードかどうかはジョブの環境変数 (PBS の `PBS_JOBID`、富士通 PJM の `PJM_JOBID`、Slurm の `SLURM_JOB_ID`) の有無で決め、
`QUICK_NODE=compute` / `QUICK_NODE=login` で強制できる。判定と使った値は最初に表示する。`OMP_NUM_THREADS` を設定していれば
その値を使う (ジョブではスケジューラが設定していることがある: PBS の `ompthreads` (キューの既定が 1 のことがある)、
富士通 PJM の `--omp thread=`)。Open MPI で `MPIEXEC_ARGS` に配置の指定 (`--bind-to`・`--map-by`) が無ければ `--bind-to none` を
足す (Open MPI は 2 プロセス以下だと各プロセスを 1 コアに固定し、複数のスレッドが 1 コアに詰め込まれるため)。

**どこで流すか**: CPU の環境はログインノードでもよい (施設の規則に従う)。GPU の環境と A64FX (Wisteria-Odyssey) は
計算ノード (短いバッチジョブか対話ジョブ) で流す (ログインノードには GPU が無く、A64FX の実行ファイルは x86 のログインノードでは
動かない)。GPU の環境で MPI の試験 (2 プロセス) を流すとき、このスクリプトはプロセスごとに GPU を 1 台ずつ割り当てる設定
(回帰試験の `gpu_bind_*.sh` のようなもの) をしないので、2 プロセスが同じ GPU を使うことがある。
富士通の `mpiexec` (`MPIEXEC=mpiexec` で、その実体が `/opt/FJSV…` にあるもの) はランクの標準出力をジョブのログに出して
check の行を読めないので、`RC_ONLY=1` (終了コード 0 なら合格) が既定になる。外すときは `RC_ONLY=0`。

```sh
export OMPI_CXX=acpp
bash examples/run_quick_checks.sh                    # 最後の行: === PASS 6 / FAIL 0 (SKIP 0)  所要 <秒> s
PARCEL_MPI=0 bash examples/run_quick_checks.sh       # MPI なし
bash examples/run_quick_checks.sh --rebuild          # 作り直す
NO_BUILD=1 bash examples/run_quick_checks.sh         # 記録を見ずに examples/build/ を実行だけ
```

### MPI なしのビルド

`-DPARCEL_MPI=0` を付け、MPI のコンパイララッパではなく SYCL コンパイラで直接ビルドする。`mpi.h` を読まず、
MPI ライブラリもリンクしない。1 プロセスで動き、`mpirun` なしで実行する。計算は MPI ありを 1 プロセスで
動かしたときと同じ (結果はビット単位で一致する)。既定 (`PARCEL_MPI` を指定しない) は MPI あり。
回帰試験での確認方法は `tests/regression/README.ja.md` の「MPI なしの構成」を参照。

```sh
acpp -O3 --acpp-targets=omp -march=native -DPARCEL_MPI=0 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas
./quick_start

# examples のサンプル (C / Fortran は gcc / gfortran)
PARCEL_MPI=0 bash examples/build_examples.sh
```

利用者のコードは MPI ありと同じ書き方で動く。違いは次のとおり。

- C++: コミュニケータは `parcel::comm comm(&q);` で作る (MPI ありの `parcel::comm comm(MPI_COMM_WORLD, &q);` に当たる)。
  `mpi_rank()` は 0、`mpi_size()` は 1。`set_crsMat(&comm, nstart, nend, n, nnz, ...)` はそのまま使えるが、
  1 プロセスが全体を持つので `nstart = 0`、`nend = n - 1` とする (違えば `std::invalid_argument`)。
  comm を取らない `set_crsMat(n, nnz, rowptr, col, val)` も使える。MPI の型を使う関数
  (`comm::mpi_comm()`、`parcel::blas(q, MPI_Comm)` など) は無い。利用者のコードの MPI の呼び出し
  (`MPI_Init` など) は `#if PARCEL_MPI` で分ける (`examples/quick_start.cpp` を参照)。
- C: ラッパと、`parcel_sycl_c.h` を include するプログラムの両方を `-DPARCEL_MPI=0` でコンパイルする。
  関数名と引数の並びは同じで、コミュニケータの引数 (`PARCEL_C_COMM`) は `int` になり使われない (0 を渡す)。
  `nstart = index_base`、`n_local = n_global` とする (違えば `PARCEL_ERR_DISTRIBUTION`)。ライブラリ内の例外は
  `MPI_Abort` せずに `PARCEL_ERR_RUNTIME` を返す。
- Fortran: `capi/parcel_sycl_mod.f90` は同じ (コミュニケータの整数は使われない)。`examples/sample_f.f90` は変えずに、
  1 プロセス用の `mpi` モジュールの代用 `examples/sample_f_mpi_stub.f90` と一緒に gfortran でコンパイルする。

### 主なコンパイル時マクロ

| マクロ | 既定 | 内容 |
|---|---|---|
| `_BLOCKING_LOOP_DOT_` | 0 | 1 で縮約 (dot / max) をブロック化する。CPU (OpenMP バックエンド) 向けで、barrier の固定費を避けて速くなる |
| `_BLOCKING_LOOP_` | 0 | 1 で全カーネルをブロック化する (A64FX 向け) |
| `_BLOCKING_LOOP_SPMV_` | 0 | 1 で SpMV もブロック化する (A64FX 向け。既定は SpMV だけ平坦なループ) |
| `PARCEL_SPMV_LADDER` | 0 | 1 で SpMV の内側ループを 4-2-1 の梯子に分割する (AMD の GPU 向け。NVIDIA の GPU では付けない。CPU では効果なし) |
| `PARCEL_PRECON_LADDER` | `PARCEL_SPMV_LADDER` | 前処理の反復ループの梯子分割 (A64FX では 0 にする) |
| `PARCEL_GEMV_AS_AXPY` | `_BLOCKING_LOOP_` の値 | 1 で gemv を axpy の繰り返しに分解する (A64FX 向け) |
| `PARCEL_A64FX_AXPY_C` | 1 | 1 で係数を値で渡す axpy と gemv の axpy 分解版を A64FX 向けに書き直した版を使う。0 で書き直す前の形 (A64FX 向け。結果は同じ。`_BLOCKING_LOOP_` のビルドだけで働く) |
| `PARCEL_CG_FUSED` | `_BLOCKING_LOOP_` の値 | 1 で `pcg` の 1 反復のカーネルを融合する (x と p の更新、r の更新と ‖r‖。A64FX 向け。結果は同じ。`_BLOCKING_LOOP_` のビルドでしか 1 にできない) |
| `PARCEL_JACOBI_NODIV_UNIT` | 1 | ヤコビ前処理の右辺の準備で、割る値 (ベクトルのスケーリングの最大値) が 1 のときは割り算をしない版を選ぶ (A64FX 向け。結果は同じ。`_BLOCKING_LOOP_` のビルドだけで働く) |
| `PARCEL_JACOBI_ROW2` | `_BLOCKING_LOOP_` の値 (`PARCEL_PRECON_LADDER` が 1 なら 0) | 1 で前処理行列が単精度のとき、ヤコビ前処理の反復を 2 行ずつ交互に処理する (A64FX 向け。結果は同じ。倍精度の行列・その場計算では使わない。`_BLOCKING_LOOP_=1` かつ `PARCEL_PRECON_LADDER=0` でしか 1 にできない) |
| `PARCEL_INT32` | 未定義 | 定義すると `PARCEL_INT` (大域の行番号・列番号など) を 32 ビットにする (既定 64 ビット) |
| `PARCEL_LOCAL_INT64` | 未定義 | 定義するとランク内の添字 `PARCEL_LINT` を 64 ビットにする (既定 32 ビット) |
| `PARCEL_FP_CONTRACT_OFF` | 未定義 | `#pragma clang fp contract(off)` を有効にする (`-ffp-contract=fast-honor-pragmas` と併用したときだけ効く。下の注意を参照) |
| `PARCEL_DD_NO_FMA` | 未定義 | double-double の積で `fma` を使わない (FMA 命令の無い環境向け) |
| `PARCEL_NO_CRS_CHECK` | 0 | 1 で `set_crsMat()` の入力検査を外す |
| `PARCEL_REQUIRE_IN_ORDER` | 1 | 0 で「キューが in-order か」の検査を外す |
| `PARCEL_ALLOW_FAST_MATH` | 未定義 | 定義すると `-ffast-math` でのビルドを許す (double-double と NaN の検出は保証しない) |
| `PARCEL_ENABLE_ALL_SOLVERS` / `PARCEL_ENABLE_<ソルバ>` | 1 | 0 でそのソルバをビルドから外す (コンパイル時間を縮めたいとき。例: `-DPARCEL_ENABLE_ALL_SOLVERS=0 -DPARCEL_ENABLE_PCG=1`) |
| `PARCEL_NBLOCK` | 12 | ブロック化カーネルの分割数の最後の既定値。実行時は環境変数 `PARCEL_NBLOCK`、次に OpenMP のスレッド数が優先される |
| `RESIDUAL_REPLACE` | 未定義 | 定義すると `pcg` / `pbicgstab` が反復ごとに真の残差も計算して表示する (診断用) |
| `PARCEL_JACOBI_ONTHEFLY` | 1 (A64FX 向けでは 0) | 1 でヤコビ前処理 (`PRECON_JACOBI`) の (D⁻¹)(L+U) の値を保存せず、前処理のたびに A の値と D⁻¹ から計算する。0 で値を保存する (A の非零 1 個あたり前処理行列の型 1 個ぶんメモリを使い、GPU では前処理が速くなる。下の「メモリと速さ」)。`_BLOCKING_LOOP_` (または `_BLOCKING_LOOP_SPMV_`) が 1 の A64FX 向けのビルドでは既定 0 |
| `PARCEL_KS_ORTHO_DEFAULT` | 未定義 (cgs2) | `ks_arnoldi` の直交化の既定 (`set_qr_algo()` を呼ばないとき): `cgs2` / `mgs`。PS-A 向けのビルドは `mgs` (1 スレッドあたりの局所長が小さく大きなキャッシュを持つ CPU では MGS2 が速いことがある)。廃止した実行時の環境変数 `PARCEL_KS_ORTHO` は読まない (設定すると rank 0 が警告を 1 行出す) |
| `PARCEL_PCG_NO_XHIST` | 1 | `pcg` の解の更新をまとめる反復数 (pstep) の既定。1 で 1 (毎反復更新)、0 で 4。実行時の環境変数 `PARCEL_PSTEP` が優先する |

実行時の環境変数:

| 環境変数 | 内容 |
|---|---|
| `PARCEL_NBLOCK` | ブロック化カーネルの分割数 |
| `PARCEL_PSTEP` | `pcg` が解の更新をまとめる反復数 (既定 1、`-DPARCEL_PCG_NO_XHIST=0` でビルドすると 4)。p の履歴を pstep 本持つ。4 にすると解の読み書きが 1/4 になって少し速くなる代わりに、ベクトル 3 本分メモリを使う (解の足し合わせ方が変わるので丸めは変わる。反復数は同じ) |
| `PARCEL_PRECON_VERBOSE` | 自動で決めた前処理の設定を表示する |
| `PARCEL_TIMER_SYNC` | 1 で内積と係数を値で渡す axpy の時間計測の前後にデバイスを同期する (時間の内訳を正確に取るとき。既定では内積の時間に直前の axpy などの時間が入る) |
| `PARCEL_MEM_REPORT` | 設定すると PARCEL が確保したデバイスメモリのピークを数える (`util_sycl::device_mem_peak_bytes()`。検証サンプルが表示する) |

## 主な既定値

| 設定 | 既定値 |
|---|---|
| `set_tolerance()` (全ソルバ) | 1.0e-8 (線形ソルバは相対残差 ‖r‖/‖b‖) |
| `set_max_iteration()` | 10000 (`ks_arnoldi` は `set_ks_max_restart()`) |
| `pgmres` のリスタート周期 `set_restart(m)` | 30 |
| `pcagmres` の `set_restart(s, t)` | s = 8, t = 4 (リスタート周期 s·t = 32) |
| `pcagmres` の Q の作り方 `set_explicit_q()` | `CAGMRES_Q_AUTO` |
| `ks_arnoldi` の `set_ks_subspace(m, k_keep, nev)` | m = 30, k_keep = 10, nev = 4 |
| `ks_arnoldi` の `set_ks_max_restart()` | 200 |
| `ks_arnoldi` の直交化 `set_qr_algo()` | `QR_CGS` (CGS2。ビルド時の `PARCEL_KS_ORTHO_DEFAULT=mgs` では `QR_MGS`) |
| 前処理 `set_precon_algo()` | `PRECON_NONE` |
| 固有値ソルバの内部線形ソルバ (`inverse_power`、`ks_arnoldi` の shift-invert) | `inverse_power` は pgmres (`set_linsolve_algo(LINSOLVE_PCG)` で pcg)、前処理はヤコビ (反復 1 回)、リスタート周期 30、収束判定値 1e-10、反復上限 5000 (`ks_arnoldi` は 2000) |
| `set_precon_spectrum()` | false (A の固有値) |

`pgmres` / `pcagmres` は、反復上限がリスタート周期の倍数でなくても上限ちょうどまで反復する
(最後の周期を短くする)。C / Fortran の `parcel_options` の既定値は `parcel_options_init()` が
ライブラリから読む。

## 使い方の最小例

`examples/quick_start.cpp` (1 次元ラプラシアンを pcg で解く) の要点:

```cpp
#include "Include/parcel_sycl.hpp"

MPI_Init(&argc, &argv);
sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};  // in-order キューが必須
parcel::comm comm(MPI_COMM_WORLD, &q);

// 各ランクが連続した行 [nstart, nend] を持つ。行ポインタは局所 (0 始まり)、列番号は大域 (0 始まり)
parcel::matrix<parcel_double> A(&q);
A.set_crsMat<double>(&comm, nstart, nend, n, nnz, rowptr, col, val);   // rowptr, col は PARCEL_INT*

parcel::vector<parcel_double> x(n, &q), b(n, &q);
x.setValue(x_host);  b.setValue(b_host);

parcel::krylov::pcg<> cg;               // 精度を省略すると倍精度
cg.set_tolerance(1.0e-9);
cg.set_precon_algo(parcel::PRECON_JACOBI);
cg.solver(A, x, b);
x.getValue(x_host);  q.wait();
if (!cg.is_converged()) { /* cg.get_converged_reason() で理由 */ }
```

固有値ソルバは `s.solver(A, eigval, eigvec)` の形で呼ぶ (`examples/example_pcg_ks.cpp`)。
C / Fortran は `examples/sample_c.c` / `examples/sample_f.f90` を参照。

## 注意

- **`set_crsMat()` は引数の配列を書き換える。** 列番号は大域番号から局所番号へ、値は各行で自ランクの列が
  先になるよう並べ替えられる (行ポインタは変わらない)。同じ配列から 2 つ目の行列 (例: A − σI) を作ると
  二重に変換されて壊れるので、行列ごとに別の配列を渡すか、呼ぶ前に写しを取っておく。
- **SYCL のキューは in-order で作る** (`sycl::property::queue::in_order{}`)。PARCEL はカーネルを event で
  つながずに順に投入するため。in-order でないキューを渡すと `std::invalid_argument` を投げる。
- **行ブロック分割と CRS の入力検査。** `set_crsMat()` は、各ランクの行ブロックが全体の行 [0, n) を隙間・重複なく
  覆うこと (ランク番号の順に並んでいなくてよい)、`n == nend-nstart+1` で 1 行以上あること、行ポインタが
  0 始まりで単調非減少・末尾が nnz であること、列番号が範囲内にあることを調べ、崩れていれば全ランクが同じ
  `std::invalid_argument` を投げる。検査を外すには `-DPARCEL_NO_CRS_CHECK=1`。
- **整数型。** 大域の量 (全体の行数、`nstart` / `nend`、`set_crsMat()` に渡す行ポインタと大域列番号) は
  `PARCEL_INT` (既定 `int64_t`、`-DPARCEL_INT32` で `int32_t`)、デバイス上の CRS の添字は `PARCEL_LINT`
  (既定 `int32_t`、`-DPARCEL_LOCAL_INT64` で `int64_t`) を使う。1 ランクの非零数 (または局所行数 + 袖の数) が
  2³¹−1 を超えると `set_crsMat()` が全ランクで `std::overflow_error` を投げるので、ランク数を増やすか
  `-DPARCEL_LOCAL_INT64` でビルドする。C / Fortran の API の整数は `PARCEL_INT` によらず 64 ビット
  (C は `parcel_int` = `int64_t`、Fortran は `integer(parcel_int_kind)`)。
- **ビルドし直して結果を一致させたい場合は `-ffp-contract=off` を付ける。** AdaptiveCpp は既定で積和を FMA に
  まとめ (`-ffp-contract=fast`。A64FX では推奨のフラグで明示する)、どこをまとめるかがビルドごとに変わりうる (特に HIP)。同じ実行ファイルなら
  結果は毎回同じだが、作り直すと丸めが変わり、低精度の構成では反復数が変わることがある。
- **`-ffast-math` は使わない。** double-double の誤差項と NaN / Inf の検出が消えるので、ライブラリはこのフラグでの
  ビルドを `#error` で止める。
- デバイスメモリの確保に失敗すると `util_sycl::device_bad_alloc` (`std::bad_alloc` の派生) を投げる。
  失敗は一部のランクでだけ起こりうるので、捕まえて続行せず全体を止めること。
- ブロックヤコビ前処理は適用の並列度がブロック数に等しく、GPU では遅い。GPU では多くの場合ヤコビ前処理の方が速い。
- ヤコビ前処理 (`PRECON_JACOBI`) は行列 A の行ポインタ・列番号を複製せずに共有し、既定 (`PARCEL_JACOBI_ONTHEFLY=1`。A64FX 向けのビルドを除く) では
  A の値も前処理のたびに読む。前処理は `solver()` の中で作られ、`solver()` から戻るときに破棄されるので、`solver()` の実行中に
  A を破棄・変更しないこと (`solver()` を呼ぶたびに作り直すので、呼び出しの間に A の値を変えるのは構わない)。
  前処理のオブジェクト (`parcel::precon_matrix`) を自分で作って使う場合は、それを使い終えるまで元の行列を破棄・変更しない。

## メモリと速さ

既定はメモリを少なくする設定で、次の 2 つでメモリを使って速くできる (どちらもアルゴリズムは同じ)。

| 設定 | 増えるメモリ (1 ランクあたり) | 効果 |
|---|---|---|
| `-DPARCEL_JACOBI_ONTHEFLY=0` (ビルド時) | ヤコビ前処理 (`PRECON_JACOBI`) の値: A の非零の数 × 前処理行列の型 (倍精度なら 8 バイト) | 前処理で掛け算と対角の判定をしない。GPU で前処理が速くなる (CPU ではほぼ変わらない) |
| `PARCEL_PSTEP=4` (実行時) または `-DPARCEL_PCG_NO_XHIST=0` (ビルド時) | `pcg` の p の履歴: 局所長 (袖を含む) のベクトル 3 本 | 解の読み書きが 1/4 になる。`pcg` が数 % 速くなる |

7 点差分の行列 (1 行に非零 7 個) では、前者がベクトル約 7 本ぶん、後者が 3 本ぶんにあたる。
A64FX 向けのビルド (`_BLOCKING_LOOP_` または `_BLOCKING_LOOP_SPMV_` が 1) では前者が既定で有効 (`PARCEL_JACOBI_ONTHEFLY` の既定 0) で、
倍精度なら A の非零 1 個あたり 8 バイト多く使う。A64FX ではその場の計算が前処理の時間に表れ、1 ノード (NX=256、pcg + ヤコビ前処理) で
値を保存すると前処理が 0.77 倍、収束までの総時間が 0.81 倍になるため。メモリを減らすなら `-DPARCEL_JACOBI_ONTHEFLY=1` を明示する。
前処理の行列を単精度にする混合精度 (`-Dparcel_solver_data_precon_mat_type=parcel_float`) は、既定のその場計算では
前処理のたびに A の倍精度の値を読むので速くなりにくい。`-DPARCEL_JACOBI_ONTHEFLY=0` と組み合わせると単精度の値を読む
(増えるメモリは A の非零 1 個あたり 4 バイト)。性能試験の混合精度 (`tests/perf/gen_perf_prec.py`) はこの組み合わせで測る。
前者は `-ffp-contract=off` なら結果がビット単位で変わらない (積和の融合を許すビルドでは丸めが変わりうる)。
後者は解の足し合わせ方が変わるので、丸めの桁で結果が変わる (反復数は同じ)。

## spack によるインストール

`spack/` に spack のレシピ (`parcel-sycl`: ヘッダと C / Fortran の API、`py-parcel-sycl`: Python のパッケージ) と、
spack の取得から LLVM・AdaptiveCpp・OpenBLAS・Python・PARCEL までをゼロから入れる `spack/stack/install_stack.sh` がある。
MPI・CUDA / ROCm・コンパイラは各計算機の既存のものを使う (`spack/stack/envs/<環境>/spack.yaml` に書く)。

```sh
bash spack/stack/install_stack.sh s86 $HOME/parcel_stack 16      # <環境> <入れ先> [並列数]
bash spack/stack/check_stack.sh $HOME/parcel_stack $HOME/parcel_check
. $HOME/parcel_stack/setup.sh
spack test run parcel-sycl py-parcel-sycl      # 入れたものでクイックスタートを動かす (GPU の版は GPU のあるノードで)
spack load parcel-sycl+mpi+cuda py-parcel-sycl+mpi+cuda
mpicxx $PARCEL_CXXFLAGS -O3 -o prog prog.cpp $PARCEL_LDFLAGS      # 利用者のプログラム (同じターゲット・最適化で)
```

同梱の環境の定義 (s86・Miyabi-C / G・PS-A / B・Wisteria-Aquarius / Odyssey) と注意は [spack/README.ja.md](spack/README.ja.md)。

## Python

`python/` は Python のパッケージ `parcel` (倍精度、MPI なし / あり)。ネイティブ層 (`python/src/parcel_py.cpp`) が
PARCEL の C++ ライブラリを直接呼ぶ。spack の `py-parcel-sycl` か、`python/` での `pip install .` で入れる。

```python
import parcel
A = parcel.DistributedCSR.from_global(A_global)          # scipy の CSR 行列などから、このランクの担当行
x, info = parcel.solve(A, b[A.local_slice()], method="pcg", precond="jacobi", tol=1e-10)
vals, vecs, info = parcel.eigs(A, method="ks_arnoldi", nev=4)
```

詳しくは [python/README.ja.md](python/README.ja.md)。

## 試験

`tests/` に回帰試験と性能試験の一式がある。どちらもジョブスクリプトの**生成器**で、
環境ごとの設定 (キュー・モジュール・コンパイルフラグ・課金先) は `tests/site_config.py` にまとめてある。
同梱の設定は開発時に使った環境 (x86 CPU、A64FX、V100 / A100 / H100 / MI300A、macOS など) の例なので、
自分の環境に合わせて書き換えてから使う (`YOUR_PROJECT` / `YOUR_GROUP` / `YOUR_USER` はプレースホルダ)。

- **回帰試験** (`tests/regression/`): 検証サンプル `tests/verify/parcel_krylov_sample_verify.cpp` を
  13 の精度構成でビルドし、7 ソルバ × 行列 (ポアソン / 移流拡散 / 複素固有値を持つ移流拡散) × 前処理 4 種 ×
  MPI 1/2/4 を NX=8 (n=512) で解いて、密行列の直接法・固有値計算による参照解と比べる。
  double-double が double に落ちていないかを見る検査 (`ddcheck`) も含む。
  ```sh
  python3 tests/regression/gen_regression.py s86_cpu     # tests/regression/sub_*.sh を生成
  bash tests/regression/sub_build_s86_cpu.sh             # 実行ファイル bin/s86_cpu/ を作る
  for f in tests/regression/sub_regression_s86_cpu_*.sh; do qsub $f; done   # パッケージのルートで投入
  bash tests/regression/summarize.sh s86_cpu             # 結果 regression_results/ を数える
  ```
  C / Fortran / Python の API の試験 (`examples/sample_c.c`・`sample_f.f90`、`tests/regression/sample_py.py`) と、
  ゼロから入れた spack の環境で流す例 (`s86spack_cpu` / `s86spack_gpu`) もある。
  どの試験も実行した版 (下の「版・引用・問い合わせ」) を生成器が `Include/parcel_version.hpp` から与える値と照合する。
- **版の照合** (`tests/check_version.py`、ビルド不要): 版を書き写しているファイル (Python のパッケージ、spack のレシピ、
  `CITATION.cff`) が `Include/parcel_version.hpp` と同じかを調べる。
- **性能試験** (`tests/perf/`): NX=512 (n = 1.3 億) を収束まで (相対残差 1e-8) 解く時間を、装置数 4 / 8 / 16 で測る
  (pgmres・pcagmres は移流拡散、他はポアソン。ソルバのパラメータはライブラリの既定)。回帰試験のビルドで作った
  `bin/<環境>/verify_double` を使う。混合精度 (`precon_dataF`) を倍精度と比べる `gen_perf_prec.py` もある
  (`precon_dataF` はヤコビ前処理のその場計算なし `-DPARCEL_JACOBI_ONTHEFLY=0` で作る。上の「メモリと速さ」)。

詳しくは [tests/regression/README.ja.md](tests/regression/README.ja.md) と [tests/perf/README.ja.md](tests/perf/README.ja.md)。

## ディレクトリ構成

```
README.md, CHANGELOG.md        この文書と v1.2 の変更点 (英語版。日本語版は README.ja.md, CHANGELOG.ja.md)
CITATION.cff, SUPPORT.md       引用の情報と問い合わせの案内 (SUPPORT.ja.md が日本語版)
COPYING, COPYING.LESSER        ライセンス (GPL-3.0 / LGPL-3.0)
doc/                           マニュアル (利用者ガイド・API リファレンス) の HTML。index.html (日本語)、index_en.html (英語)
Include/                       ライブラリ本体 (ヘッダのみ)。parcel_sycl.hpp を include する
capi/                          C / Fortran の API
  parcel_sycl_c.h                C の宣言 (parcel_solve_linear / parcel_solve_eigen など)
  parcel_sycl_mod.f90            Fortran のモジュール
  wrapper_parcel_sycl.cpp        C / Fortran API の実装 (SYCL コンパイラでビルドする)
  parcel_capi_default.hpp        ラッパのコンパイル時の既定値 (内部精度・旧 API の設定)
examples/                      サンプル
  quick_start.cpp                最小の C++ の例 (pcg。厳密解との誤差を判定する)
  quick_start_c.c, quick_start_f.f90, quick_start.py
                                 同じ問題を C / Fortran の API と Python のパッケージで解く最小の例 (MPI なし / あり)
  example_pcg_ks.cpp             線形 (pcg) と固有値 (ks_arnoldi) の C++ の例
  sample_c.c, sample_f.f90       C / Fortran の API で全ソルバを呼んで検証する例
  sample_f_mpi_stub.f90          MPI なしのビルドで sample_f.f90・quick_start_f.f90 を組むための 1 プロセス用 mpi モジュール
  problem.f90                    3 次元ポアソン行列を CRS で作る Fortran のルーチン (旧 API 用)
  build_examples.sh              上のサンプルをまとめてビルドするスクリプト
  run_quick_checks.sh            ビルド後の確認: クイックスタートをビルド・実行し合否を出す
python/                        Python のパッケージ parcel (README.md、pyproject.toml、CMakeLists.txt)
  parcel/                        Python の層
  src/parcel_py.cpp, .h          ネイティブ層 (libparcel_py.so)
  envs/mac_brew.sh               macOS + Homebrew でビルドするときの設定
spack/                         spack のレシピとゼロから入れる仕組み (README.md)
  spack_repo/parcel/             レシピ parcel-sycl / py-parcel-sycl
  stack/                         install_stack.sh、check_stack.sh、環境ごとの定義 envs/<環境>/
tests/
  site_config.py                 試験の環境ごとの設定 (書き換えて使う)
  check_version.py               版の照合 (Include/parcel_version.hpp と、版を書き写しているファイル)
  verify/                        検証サンプルと、試験用の行列の生成 (ポアソン・移流拡散・非構造)
  regression/                    回帰試験の生成器・投入と集計のスクリプト・ddcheck・Python の API の試験
  perf/                          性能試験の生成器と集計のスクリプト
tools/
  make_tarball.py                配布 tarball と sha256 を作る (同じ中身から同じ tarball)
```

## 版・引用・問い合わせ

版は `Include/parcel_version.hpp` に 1 か所だけ書いてある (この版は 1.2.0、公開日 2026-10-23)。

| 言語 | 版を得る方法 |
|---|---|
| C++ | マクロ `PARCEL_VERSION_MAJOR` / `_MINOR` / `_PATCH`、`PARCEL_VERSION_STRING` (`"1.2.0"`)、`PARCEL_VERSION_NUMBER` (10200)、`PARCEL_VERSION_GE(1, 2, 0)`。関数 `parcel::version_string()`、`parcel::get_version(&major, &minor, &patch)` |
| C | `parcel_get_version(&major, &minor, &patch)`、`parcel_version_string()` (ラッパをビルドしたときのヘッダの値) |
| Fortran | `ierr = parcel_get_version(major, minor, patch)` (`integer(c_int)`)、`call parcel_version_string(str)` |
| Python | `parcel.__version__` (パッケージの版)、`parcel.library_version()` (ネイティブ層をビルドしたヘッダの版)。両方とも同じ値になる |

Python のパッケージ (`python/pyproject.toml`)・spack のレシピ (`version("1.2.0")`) と環境の定義 (`parcel-sycl@1.2.0`)・`CITATION.cff` の版は
書き写しなので、版を上げるときはそろえて書き換え、`python3 tests/check_version.py` で確かめる。

引用するときの情報は [CITATION.cff](CITATION.cff) (引用してほしい論文: Y. Idomura et al., ScalA 17 (2017))、
問い合わせの案内は [SUPPORT.ja.md](SUPPORT.ja.md)。問い合わせ先: GitHub の Issues <https://github.com/parcel-jaea/PARCEL-release/issues>

## 配布 tarball

配布する tarball は `tools/make_tarball.py` で作る。同じ中身からは同じ tarball (同じ sha256) ができる
(ファイルの順・時刻・所有者・圧縮の頭をそろえる。試験の生成物や `__pycache__` は入れない)。

```sh
python3 tools/make_tarball.py --outdir <出力先>     # parcel_sycl-1.2.0.tar.gz と parcel_sycl-1.2.0.tar.gz.sha256
sha256sum -c parcel_sycl-1.2.0.tar.gz.sha256        # 受け取った側の確認 (macOS は shasum -a 256 -c)
python3 tools/make_tarball.py --check              # spack のレシピに書いた sha256 が今のパッケージの tarball と一致するか
```

spack のレシピ (`version("1.2.0", sha256=...)`) は tarball の中にも入るので、tarball に入れるときは常に sha256 を
外した形 (`version("1.2.0")`) にしている。手元のレシピに `--pin` で sha256 を書いても tarball は変わらない。
tarball は GitHub の Release (<https://github.com/parcel-jaea/PARCEL-release/releases>、タグ `v1.2.0`) に置き、レシピの `url` はその場所を指す。
tarball から spack で入れる手順は [spack/README.ja.md](spack/README.ja.md) の「配布 tarball から入れる」。

## ライセンス

GNU Lesser General Public License version 3 (LGPL-3.0-only)。[COPYING.LESSER](COPYING.LESSER) と
[COPYING](COPYING) を参照。各ソースファイルの先頭に `SPDX-License-Identifier: LGPL-3.0-only` と著作権の行がある。

Copyright (c) 日本原子力研究開発機構 システム計算科学センター HPC・DX基盤技術開発室

配布元: <https://ccse.jaea.go.jp/software/PARCEL/>、公開用リポジトリ (配布 tarball の Release): <https://github.com/parcel-jaea/PARCEL-release>
