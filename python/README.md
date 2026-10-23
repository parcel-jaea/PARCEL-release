English | [日本語](README.ja.md)

# parcel (Python Interface of PARCEL_SYCL)

A package for calling PARCEL_SYCL from Python. A native layer (`src/parcel_py.cpp` → `libparcel_py.so`), independent of the C / Fortran API (`capi/parcel_sycl_c.h`),
calls the PARCEL C++ library (`../Include/`) directly.

- A single call does everything from registering the matrix to solving. Double precision only.
- The default is the single-process version without MPI (`python script.py`). Building with `PARCEL_MPI=1` gives the version with MPI,
  which is launched as `mpirun -n 4 python script.py`, with all ranks making the same calls. mpi4py is optional (MPI_COMM_WORLD if absent).
- Unspecified settings use the library defaults (the Python side has no defaults of its own).
- Requirements: numpy (scipy is convenient for building matrices; the regression test `tests/regression/sample_py.py` uses scipy).
- The version is the same as the library itself (`parcel.__version__`; this version is 1.2.0). `parcel.library_version()` returns the version of the headers
  (`Include/parcel_version.hpp`) the native layer was built with, which is the same as `__version__`.

```python
import parcel
A = parcel.DistributedCSR.from_global(A_global)          # rows assigned to this rank (all rows without MPI)
x, info = parcel.solve(A, b[A.local_slice()], method="pcg", precond="jacobi", tol=1e-10)
vals, vecs, info = parcel.eigs(A, method="ks_arnoldi", m=40, k_keep=10, nev=4)
print(parcel.build_info(), parcel.has_mpi(), parcel.device_name())
```

## Installation

The native layer is built for each environment (SYCL target, MPI). AdaptiveCpp and LAPACKE (OpenBLAS etc.) are required.

### spack

Install with the recipe `py-parcel-sycl` in `spack/` (`spack/README.md`). The from-scratch installer `spack/stack/install_stack.sh`
installs this package together with the C++ headers and the C / Fortran APIs.

```bash
. <prefix>/setup.sh
spack load py-parcel-sycl~mpi~cuda~rocm      # load by specifying the configuration
python -c "import parcel; print(parcel.build_info())"
```

### pip

Run `pip install .` in `python/` (uses scikit-build-core and CMake). Settings are given with `-C cmake.define.<name>=<value>`
(the list is at the top of `CMakeLists.txt`).

| Name | Default | Meaning |
|---|---|---|
| `PARCEL_TARGET` | `omp` | AdaptiveCpp target (`cuda:sm_70`, `cuda:sm_90`, `hip:gfx942`, etc.) |
| `PARCEL_KERNEL_FLAGS` | Determined from the target | Kernel optimization macros (CPU: `-D_BLOCKING_LOOP_DOT_=1`; AMD GPU (`hip:`): `-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1`; NVIDIA GPU (`cuda:`): no ladder splitting) |
| `PARCEL_HOST_FLAGS` | `-march=native -mtune=native` | Optimization for the host |
| `PARCEL_EXTRA_FLAGS` | None | Other flags (`-Xcuda-ptxas -maxrregcount=255` is recommended on V100) |
| `PARCEL_INCLUDE_DIR` | `../Include` | PARCEL headers |
| `LAPACKE_INCLUDE_DIR` / `LAPACK_LIBS` | `/usr/include/openblas` / `-lopenblas` | LAPACKE |
| `PARCEL_MPI` | `0` | `1` for MPI (the C++ compiler defaults to `mpicxx`; set `OMPI_CXX=acpp` etc. beforehand) |
| `PARCEL_MPICXX` | `mpicxx` with MPI, `acpp` without | C++ compiler |

```bash
cd python
pip install . -C cmake.define.PARCEL_TARGET=cuda:sm_70 \
              -C "cmake.define.PARCEL_EXTRA_FLAGS=-Xcuda-ptxas -maxrregcount=255"
```

At run time, use the same environment as for the build (an environment where the AdaptiveCpp, LAPACK and MPI libraries are found).

### Without pip

Build the native layer directly, and give its location with `PARCEL_PY_LIB` and `python/` with `PYTHONPATH`
(the Python jobs of the regression tests take this form; `tests/regression/sub_build_lang_<env>.sh` makes `bin/<env>/libparcel_py.so`).

```bash
acpp -O3 --acpp-targets=omp -DPARCEL_MPI=0 -march=native -mtune=native -D_BLOCKING_LOOP_DOT_=1 \
     -fPIC -shared -IInclude -I/usr/include/openblas -o libparcel_py.so python/src/parcel_py.cpp -lopenblas
PYTHONPATH=python PARCEL_PY_LIB=$PWD/libparcel_py.so python script.py
```

### macOS (Homebrew)

After `brew install adaptivecpp open-mpi openblas gcc numpy scipy`, source the build settings (`PARCEL_TARGET=omp.library-only`, `-D_BLOCKING_LOOP_DOT_=1`, Homebrew OpenBLAS)
with `. python/envs/mac_brew.sh`, and then do the pip or direct build above.
spack is not used on macOS.

## Tests

`tests/regression/sample_py.py` (which does the same problems, items and judgments as the C `examples/sample_c.c` from Python) can be
run as the language `python` of the regression tests (the "Tests of the C / Fortran / Python APIs" section of `tests/regression/README.md`).
It also runs standalone.

```bash
PARCEL_TEST_FROM_SOURCE=1 PARCEL_PY_LIB=$PWD/libparcel_py.so python tests/regression/sample_py.py 16   # reads python/ of the package
python tests/regression/sample_py.py 16                                                                # reads the installed parcel
```
