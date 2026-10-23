[English](README.md) | 日本語

# parcel (PARCEL_SYCL の Python インタフェース)

PARCEL_SYCL を Python から呼ぶためのパッケージ。C / Fortran の API (`capi/parcel_sycl_c.h`) とは独立した
ネイティブ層 (`src/parcel_py.cpp` → `libparcel_py.so`) が、PARCEL の C++ ライブラリ (`../Include/`) を直接呼ぶ。

- 1 回の呼び出しで行列の登録から解法までを行う。精度は倍精度だけ。
- 既定は MPI なしの 1 プロセス版 (`python script.py`)。`PARCEL_MPI=1` でビルドすると MPI ありの版になり、
  `mpirun -n 4 python script.py` のように起動して全ランクが同じ呼び出しを行う。mpi4py は任意 (無ければ MPI_COMM_WORLD)。
- 未指定の設定はライブラリの既定値を使う (Python 側は既定値を持たない)。
- 必要なもの: numpy (scipy は行列を作るのに便利。回帰試験の `tests/regression/sample_py.py` は scipy を使う)。
- 版はライブラリ本体と同じ (`parcel.__version__`、この版は 1.2.0)。`parcel.library_version()` はネイティブ層を
  ビルドしたヘッダ (`Include/parcel_version.hpp`) の版を返し、`__version__` と同じになる。

```python
import parcel
A = parcel.DistributedCSR.from_global(A_global)          # このランクの担当行 (MPI なしでは全行)
x, info = parcel.solve(A, b[A.local_slice()], method="pcg", precond="jacobi", tol=1e-10)
vals, vecs, info = parcel.eigs(A, method="ks_arnoldi", m=40, k_keep=10, nev=4)
print(parcel.build_info(), parcel.has_mpi(), parcel.device_name())
```

## インストール

ネイティブ層は環境 (SYCL のターゲット・MPI) ごとにビルドする。AdaptiveCpp と LAPACKE (OpenBLAS など) が要る。

### spack

`spack/` のレシピ `py-parcel-sycl` で入れる (`spack/README.ja.md`)。ゼロから入れる `spack/stack/install_stack.sh` は
C++ のヘッダ・C / Fortran の API と一緒にこのパッケージも入れる。

```bash
. <入れ先>/setup.sh
spack load py-parcel-sycl~mpi~cuda~rocm      # 構成を指定して読む
python -c "import parcel; print(parcel.build_info())"
```

### pip

`python/` で `pip install .` する (scikit-build-core と CMake を使う)。設定は `-C cmake.define.<名前>=<値>` で与える
(`CMakeLists.txt` の先頭に一覧がある)。

| 名前 | 既定 | 意味 |
|---|---|---|
| `PARCEL_TARGET` | `omp` | AdaptiveCpp のターゲット (`cuda:sm_70`、`cuda:sm_90`、`hip:gfx942` など) |
| `PARCEL_KERNEL_FLAGS` | ターゲットから決める | カーネル最適化のマクロ (CPU は `-D_BLOCKING_LOOP_DOT_=1`、AMD の GPU (`hip:`) は `-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1`。NVIDIA の GPU (`cuda:`) は梯子分割なし) |
| `PARCEL_HOST_FLAGS` | `-march=native -mtune=native` | ホスト向けの最適化 |
| `PARCEL_EXTRA_FLAGS` | なし | そのほかのフラグ (V100 では `-Xcuda-ptxas -maxrregcount=255` を推奨) |
| `PARCEL_INCLUDE_DIR` | `../Include` | PARCEL のヘッダ |
| `LAPACKE_INCLUDE_DIR` / `LAPACK_LIBS` | `/usr/include/openblas` / `-lopenblas` | LAPACKE |
| `PARCEL_MPI` | `0` | `1` で MPI あり (C++ コンパイラは既定で `mpicxx`。`OMPI_CXX=acpp` などを設定しておく) |
| `PARCEL_MPICXX` | MPI ありは `mpicxx`、なしは `acpp` | C++ コンパイラ |

```bash
cd python
pip install . -C cmake.define.PARCEL_TARGET=cuda:sm_70 \
              -C "cmake.define.PARCEL_EXTRA_FLAGS=-Xcuda-ptxas -maxrregcount=255"
```

実行時もビルドと同じ環境 (AdaptiveCpp・LAPACK・MPI のライブラリが見つかる環境) にする。

### pip を使わない

ネイティブ層を直接ビルドし、`PARCEL_PY_LIB` でその場所を、`PYTHONPATH` で `python/` を与える
(回帰試験の Python のジョブはこの形。`tests/regression/sub_build_lang_<環境>.sh` が `bin/<環境>/libparcel_py.so` を作る)。

```bash
acpp -O3 --acpp-targets=omp -DPARCEL_MPI=0 -march=native -mtune=native -D_BLOCKING_LOOP_DOT_=1 \
     -fPIC -shared -IInclude -I/usr/include/openblas -o libparcel_py.so python/src/parcel_py.cpp -lopenblas
PYTHONPATH=python PARCEL_PY_LIB=$PWD/libparcel_py.so python script.py
```

### macOS (Homebrew)

`brew install adaptivecpp open-mpi openblas gcc numpy scipy` の後、`. python/envs/mac_brew.sh` で
ビルドの設定 (`PARCEL_TARGET=omp.library-only`、`-D_BLOCKING_LOOP_DOT_=1`、Homebrew の OpenBLAS) を読み込んでから
上の pip または直接ビルドを行う。macOS では spack は使わない。

## 試験

`tests/regression/sample_py.py` (C の `examples/sample_c.c` と同じ問題・項目・判定を Python から行う) を
回帰試験の言語 `python` として流せる (`tests/regression/README.ja.md` の「C / Fortran / Python の API の試験」)。
単独でも動く。

```bash
PARCEL_TEST_FROM_SOURCE=1 PARCEL_PY_LIB=$PWD/libparcel_py.so python tests/regression/sample_py.py 16   # パッケージの python/ を読む
python tests/regression/sample_py.py 16                                                                # インストールした parcel を読む
```
