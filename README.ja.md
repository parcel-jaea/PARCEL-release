[English](README.md) | 日本語

# PARCEL_SYCL

**PARCEL** (PARallel Computing ELements) は、SYCL で書かれた疎行列向けの並列 Krylov 部分空間法ソルバのライブラリである。
CPU (OpenMP バックエンド) と NVIDIA / AMD の GPU で動き、プロセス間は MPI で並列化する。
演算性能が高い一方でノード間通信が相対的に遅い並列計算機に向けて設計しており、
混合精度 (半精度・単精度・倍精度・double-double) に対応する。

| | |
|---|---|
| **マニュアル (HTML)** | **<https://ccse2.jaea.go.jp/software/parcel/index.html>** ([English](https://ccse2.jaea.go.jp/software/parcel/index_en.html)) — tarball の `doc/` にも同梱 |
| ダウンロード | [最新リリース (v1.2.0)](https://github.com/parcel-jaea/PARCEL-release/releases/latest) |
| ライセンス | LGPL-3.0 |
| 問い合わせ | [GitHub Issues](https://github.com/parcel-jaea/PARCEL-release/issues) |

## 提供ソルバ

| 種類 | ソルバ | 問題 |
|---|---|---|
| 線形 | `pcg` | 対称正定値行列。最も軽い |
| | `pbicgstab` | 非対称行列。1 反復を軽くしたい場合 |
| | `pgmres` | 非対称行列 (または対称だが正定値でない行列)。収束を安定させたい場合 |
| | `pcagmres` | `pgmres` と同じ問題で、ランク数が多く集団通信が律速する場合 (通信削減型) |
| 固有値 | `power` | 絶対値最大の固有値 1 個 |
| | `inverse_power` | 絶対値最小の固有値 1 個 |
| | `ks_arnoldi` | 複数の固有値、複素共役対の固有値、シフトに近い固有値 (shift-invert) |
| 前処理 | ポイント Jacobi | 最も安価。効果は小さい |
| | ブロック Jacobi | CPU 向け。作用はランク数と分割数で変わる |
| | Jacobi | GPU 向け。作用は分割によらない |

API: C++ (ヘッダだけのライブラリ)、C言語・Fortran (API のライブラリ)、Python (パッケージ `parcel`)。

## クイックスタート

必要なもの: SYCL の実装 ([AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp))、LAPACKE (OpenBLAS など)、
MPI 版では MPI ライブラリ。入れ方 (spack でゼロから、既存の spack へ、macOS は Homebrew) は
[SYCL 対応コンパイラの導入](https://ccse2.jaea.go.jp/software/parcel/latest/ja/user/quickstart.html#qs-compiler) にある。

`examples/quick_start.cpp` は 1 次元ラプラシアン (三重対角、n = 1000、b = 1) を共役勾配法で解き、
厳密解と照合する。CPU (OpenMP バックエンド) では、パッケージのトップディレクトリで次のとおり。

```sh
# MPI なし (1 プロセス)
acpp -O3 --acpp-targets=omp -march=native -DPARCEL_MPI=0 -I/usr/include/openblas \
     -o quick_start examples/quick_start.cpp -lopenblas
OMP_NUM_THREADS=8 ./quick_start

# MPI あり (Open MPI)
export OMPI_CXX=acpp
mpicxx -O3 --acpp-targets=omp -march=native -I/usr/include/openblas \
       -o quick_start examples/quick_start.cpp -lopenblas
OMP_NUM_THREADS=2 mpirun -n 4 ./quick_start
```

```
 converged_reason = 1 (CONVERGED_RTOL) , its = 500 , residual = 1.161602e-15 , true_residual = 1.183245e-13
reason = 1 (CONVERGED_RTOL), iterations = 500, residual = 1.162e-15, true residual = 1.183e-13
x[0] = 500.000000 (exact 500.000000)
check: max relative error = 1.252e-15 (limit 1.0e-06) PASS
```

NVIDIA の GPU ではターゲットだけを変える (例: `--acpp-targets=cuda:sm_70`)。プログラムの要点は次のとおり。

```cpp
sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};   // in-order 必須
parcel::comm comm(MPI_COMM_WORLD, &q);                // MPI なし: parcel::comm comm(&q);

parcel::matrix<> A(&q);                               // <> は倍精度
A.set_crsMat<double>(&comm, nstart, nend, n, nnz, rowptr, col, val);   // 自ランクの担当行の CRS
parcel::vector<> x(n, &q), b(n, &q);
x.setValue(x_h);  b.setValue(b_h);

parcel::krylov::pcg<> cg;
cg.set_tolerance(1.0e-9);                             // 相対残差 ||r|| / ||b|| (既定 1e-8)
cg.solver(A, x, b);
x.getValue(x_h);  q.wait();
if (!cg.is_converged()) { /* cg.get_converged_reason() */ }
```

C言語・Fortran・Python で同じ問題を解く例は [クイックスタート](https://ccse2.jaea.go.jp/software/parcel/latest/ja/user/quickstart.html) にある。
`bash examples/run_quick_checks.sh` でこれらをまとめてビルド・実行し、`PASS n / FAIL m` を表示する。

## 基本構成

- **1 MPI ランク = 1 デバイス** (GPU 1 基、MIG の 1 区画、APU の GPU 部、または CPU コアの 1 グループ)。
  ランク間の通信は MPI が、ランク内の並列化は SYCL が受け持つ。
- **行ブロック分割**: 各ランクは行列とベクトルの連続した行 `nstart`〜`nend` を受け持つ。
  全ランクの行ブロックで全行を隙間も重なりもなく覆う必要がある。
- **CRS 入力**: 行ポインタはローカル (担当の先頭行を 0 とする)、列番号はグローバル。
  `set_crsMat()` は列番号を付け替えてハロ交換の表を作る。このとき呼び出し側の列番号と値の配列を書き換える。
- **ハロ交換** (疎行列ベクトル積の前に毎回行う) はホストのバッファを経由するので、GPU 対応 MPI は要らない。
- **精度**: データの精度と演算の精度を別々に選べる (`parcel_fp16`、`parcel_float`、`parcel_double`、
  `parcel_doubledouble`)。省略した精度は倍精度になる。

詳しくは: [基本構成](https://ccse2.jaea.go.jp/software/parcel/latest/ja/user/concepts.html) /
[ソルバの選び方](https://ccse2.jaea.go.jp/software/parcel/latest/ja/user/choosing-solver.html) /
[API リファレンス](https://ccse2.jaea.go.jp/software/parcel/latest/ja/reference/index.html)

## 動作確認環境

Intel と Arm の CPU (Xeon、A64FX、Apple M4) と NVIDIA / AMD の GPU (V100、A100、GH200、RTX A6000、MI300A)、
AdaptiveCpp 24.10 (macOS は 25.10)。OS・コンパイラ・MPI を含む一覧と、PETSc との性能比較は
[マニュアルのトップページ](https://ccse2.jaea.go.jp/software/parcel/index.html) にある。

## 改変履歴

| 版 | 公開日 | |
|---|---|---|
| [v1.2.0](https://github.com/parcel-jaea/PARCEL-release/releases/tag/v1.2.0) | 2026-10-23 | 固有値ソルバ (`power`、`inverse_power`、`ks_arnoldi`)、C言語・Fortran から全 7 ソルバを呼ぶ API と Python パッケージ、MPI なしのビルド、spack のレシピと macOS の Homebrew、環境ごとのカーネル最適化、13 精度構成での検証 |
| [v1.0.0](https://github.com/parcel-jaea/PARCEL-release/releases/tag/v1.0.0) | 2025-06 | 最初の SYCL 版 |

変更の詳細は [CHANGELOG.ja.md](CHANGELOG.ja.md)。SYCL 版より前の PARCEL 1.0〜1.2 (Fortran / CUDA) は [PARCEL-legacy](https://github.com/parcel-jaea/PARCEL-legacy) にある。

## 引用・開発者・ライセンス

- 引用: [CITATION.cff](CITATION.cff)
- 開発者: 日本原子力研究開発機構 システム計算科学センター
- ライセンス: GNU Lesser General Public License v3.0 ([COPYING.LESSER](COPYING.LESSER)、[COPYING](COPYING))
