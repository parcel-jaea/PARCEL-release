English | [日本語](README.ja.md)

# PARCEL_SYCL

**PARCEL** (PARallel Computing ELements) is a library of parallel Krylov subspace solvers for sparse matrices,
written in SYCL. It runs on CPUs (OpenMP backend) and on NVIDIA / AMD GPUs, with MPI parallelism across processes.
It is designed for parallel computers whose computing performance is high but whose inter-node communication is relatively slow,
and supports mixed precision (half, single, double, double-double).

| | |
|---|---|
| **Manual (HTML)** | **<https://ccse2.jaea.go.jp/software/parcel/index_en.html>** ([日本語](https://ccse2.jaea.go.jp/software/parcel/index.html)) — also included in the tarball as `doc/` |
| Download | [Latest release (v1.2.0)](https://github.com/parcel-jaea/PARCEL-release/releases/latest) |
| License | LGPL-3.0 |
| Contact | [GitHub Issues](https://github.com/parcel-jaea/PARCEL-release/issues) |

## Solvers

| Kind | Solver | Problem |
|---|---|---|
| Linear | `pcg` | Symmetric positive definite matrices; the lightest |
| | `pbicgstab` | Nonsymmetric matrices, when each iteration should be light |
| | `pgmres` | Nonsymmetric (or symmetric indefinite) matrices, when stable convergence is wanted |
| | `pcagmres` | Same as `pgmres`, when many ranks make collective communication the bottleneck (communication-avoiding) |
| Eigenvalue | `power` | One eigenvalue of largest magnitude |
| | `inverse_power` | One eigenvalue of smallest magnitude |
| | `ks_arnoldi` | Several eigenvalues, complex conjugate pairs, eigenvalues near a shift (shift-invert) |
| Preconditioner | Point Jacobi | The cheapest; small effect |
| | Block Jacobi | For CPUs; its effect depends on the numbers of ranks and blocks |
| | Jacobi | For GPUs; its effect does not depend on the partitioning |

APIs: C++ (header-only), C and Fortran (API library), Python (package `parcel`).

## Quick Start

Requirements: a SYCL implementation ([AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)), LAPACKE (e.g. OpenBLAS),
and an MPI library for the MPI version. How to install them (spack from scratch, an existing spack, Homebrew on macOS) is in
[Installing a SYCL Compiler](https://ccse2.jaea.go.jp/software/parcel/latest/en/user/quickstart.html#qs-compiler).

`examples/quick_start.cpp` solves the 1D Laplacian (tridiagonal, n = 1000, b = 1) with the conjugate gradient method
and checks the result against the exact solution. On a CPU (OpenMP backend), in the top directory of the package:

```sh
# Without MPI (one process)
acpp -O3 --acpp-targets=omp -march=native -DPARCEL_MPI=0 -I/usr/include/openblas \
     -o quick_start examples/quick_start.cpp -lopenblas
OMP_NUM_THREADS=8 ./quick_start

# With MPI (Open MPI)
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

For an NVIDIA GPU change only the target (e.g. `--acpp-targets=cuda:sm_70`). The core of the program:

```cpp
sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};   // must be in-order
parcel::comm comm(MPI_COMM_WORLD, &q);                // without MPI: parcel::comm comm(&q);

parcel::matrix<> A(&q);                               // <> = double precision
A.set_crsMat<double>(&comm, nstart, nend, n, nnz, rowptr, col, val);   // CRS of the rows owned by this rank
parcel::vector<> x(n, &q), b(n, &q);
x.setValue(x_h);  b.setValue(b_h);

parcel::krylov::pcg<> cg;
cg.set_tolerance(1.0e-9);                             // relative residual ||r|| / ||b|| (default 1e-8)
cg.solver(A, x, b);
x.getValue(x_h);  q.wait();
if (!cg.is_converged()) { /* cg.get_converged_reason() */ }
```

The same problem in C, Fortran and Python: [Quick Start](https://ccse2.jaea.go.jp/software/parcel/latest/en/user/quickstart.html).
`bash examples/run_quick_checks.sh` builds and runs all of them and prints `PASS n / FAIL m`.

## Basic Structure

- **1 MPI rank = 1 device** (one GPU, one MIG partition, the GPU part of an APU, or a group of CPU cores).
  MPI handles communication between ranks; SYCL handles parallelism within a rank.
- **Row-block partitioning**: each rank owns a contiguous block of rows `nstart`–`nend` of the matrix and the vectors.
  The blocks of all ranks must cover all rows without gaps or overlaps.
- **CRS input**: the row pointers are local (0-based from the first owned row) and the column indices are global.
  `set_crsMat()` renumbers the columns and builds the halo-exchange table; it overwrites the caller's column and value arrays.
- **Halo exchange** before each sparse matrix-vector product goes through host buffers, so a GPU-aware MPI is not required.
- **Precision**: the data precision and the computation precision can be chosen separately (`parcel_fp16`, `parcel_float`,
  `parcel_double`, `parcel_doubledouble`); omitted precisions are double.

Details: [Basic Structure](https://ccse2.jaea.go.jp/software/parcel/latest/en/user/concepts.html) /
[Choosing a Solver](https://ccse2.jaea.go.jp/software/parcel/latest/en/user/choosing-solver.html) /
[API Reference](https://ccse2.jaea.go.jp/software/parcel/latest/en/reference/index.html)

## Tested Environments

Intel and Arm CPUs (Xeon, A64FX, Apple M4) and NVIDIA / AMD GPUs (V100, A100, GH200, RTX A6000, MI300A),
with AdaptiveCpp 24.10 (25.10 on macOS). The list with OS, compilers and MPI, and performance comparisons with PETSc, are on the
[manual top page](https://ccse2.jaea.go.jp/software/parcel/index_en.html).

## Release History

| Version | Date | |
|---|---|---|
| [v1.2.0](https://github.com/parcel-jaea/PARCEL-release/releases/tag/v1.2.0) | 2026-10-23 | Eigenvalue solvers (`power`, `inverse_power`, `ks_arnoldi`); C / Fortran API for all 7 solvers and the Python package; build without MPI; spack recipes and Homebrew on macOS; kernel optimizations per environment; verification with 13 precision configurations |
| [v1.0.0](https://github.com/parcel-jaea/PARCEL-release/releases/tag/v1.0.0) | 2025-06 | First SYCL version |

Details of the changes: [CHANGELOG.md](CHANGELOG.md). The versions before SYCL (PARCEL 1.0–1.2, Fortran / CUDA) are in [PARCEL-legacy](https://github.com/parcel-jaea/PARCEL-legacy).

## Citation, Developer and License

- Citation: [CITATION.cff](CITATION.cff)
- Developer: Center for Computational Science & e-Systems, Japan Atomic Energy Agency
- License: GNU Lesser General Public License v3.0 ([COPYING.LESSER](COPYING.LESSER), [COPYING](COPYING))
