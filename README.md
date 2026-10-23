English | [日本語](README.ja.md)

# PARCEL_SYCL v1.2

PARCEL_SYCL is a library of parallel Krylov subspace methods written in SYCL (PARCEL: Parallel Computing Elements).
It solves systems of linear equations and eigenvalue problems for sparse matrices (CRS format) with distributed parallelism by MPI and on SYCL devices
(the OpenMP backend on CPUs, NVIDIA / AMD GPUs, and so on). The library itself consists of headers only.

v1.2 adds the following to v1.1 (see [CHANGELOG.md](CHANGELOG.md) for details).

- Added the eigenvalue solvers `ks_arnoldi` (Arnoldi + Krylov-Schur restart) and `inverse_power` (inverse power method).
  `pcaarnoldi` of v1.1 has been removed.
- An API for calling all 7 solvers from C / Fortran (`parcel_solve_linear` / `parcel_solve_eigen`).
- An API that returns the convergence state (`get_converged_reason()` and others), and input checks (row-block partitioning, CRS, arguments).
- The default integer type for row numbers and the like is now 64 bits, and only the CRS indices on the device are 32 bits (`PARCEL_INT` / `PARCEL_LINT`).
- The default settings were changed to values that solve problems without specifying anything (tolerance 1e-8, maximum number of iterations 10000, and so on).
- Bug fixes and performance improvements on CPUs and GPUs. A full set of regression tests and performance tests.

The manual (User Guide and API Reference) is included as HTML in `doc/` (English [doc/index_en.html](doc/index_en.html),
Japanese [doc/index.html](doc/index.html)). Open it directly in a web browser; it does not connect to external sites.

Contents: [Solvers](#solvers) / [Precision](#precision) / [Requirements](#requirements) / [Build](#build) / [Main Defaults](#main-defaults) /
[Minimal Example](#minimal-example) / [Notes](#notes) / [Memory and Speed](#memory-and-speed) /
[Installing with spack](#installing-with-spack) / [Python](#python) / [Tests](#tests) / [Directory Layout](#directory-layout) /
[Version, Citation and Contact](#version-citation-and-contact) / [Release Tarball](#release-tarball) / [License](#license)

## Solvers

| Type | Class (`parcel::krylov::`) | What it computes |
|---|---|---|
| Linear | `pcg` | Preconditioned conjugate gradient method (A is symmetric positive definite) |
| Linear | `pbicgstab` | Preconditioned BiCGStab method |
| Linear | `pgmres` | Preconditioned GMRES method (with restart) |
| Linear | `pcagmres` | Preconditioned communication-avoiding GMRES method (s-step, with restart) |
| Eigenvalue | `power` | Power method. One eigenvalue of largest magnitude |
| Eigenvalue | `inverse_power` | Inverse power method. One eigenvalue of smallest magnitude (solves linear equations internally) |
| Eigenvalue | `ks_arnoldi` | Arnoldi + Krylov-Schur. `nev` eigenvalues on the largest-magnitude side, or `nev` eigenvalues near σ with `set_shift(σ)` (shift-invert) |

- `ks_arnoldi` can also compute complex conjugate pairs of eigenvalues. `power` / `inverse_power` use a real Rayleigh quotient, so they
  do not converge to complex eigenvalues.
- `pcagmres` lets you choose with `set_explicit_q()` whether to form the orthogonal basis Q explicitly
  (`CAGMRES_Q_AUTO` (default) / `CAGMRES_Q_EXPLICIT` / `CAGMRES_Q_IMPLICIT`).
  `CAGMRES_Q_IMPLICIT` is the CholeskyQR version that does not form Q, and corresponds to `cholQR_pcagmres` of v1.1.
  The default QR (`set_qr_algo()`) is `QR_MGS` (`QR_CHOLESKYQR` with `CAGMRES_Q_IMPLICIT` if not specified).
- By default the eigenvalue solvers return the eigenvalues of A. The eigenvalues of the preconditioned matrix A K⁻¹ are computed with `set_precon_spectrum(true)`.
- The orthogonalization of `ks_arnoldi` is selected with `set_qr_algo()` (the same constants as `pgmres`; in `ks_arnoldi` both are applied twice).
  `QR_CGS` is block classical Gram-Schmidt applied twice (CGS2, default), and `QR_MGS` is modified Gram-Schmidt applied twice (MGS2).
  In CPU configurations with a small local length per rank MGS2 can be faster, while on CPUs with a large local length and on GPUs CGS2 is faster.
  For C / Fortran use `parcel_options.ks_qr_algo`, and for Python `eigs(..., orthogonalization="cgs" | "mgs")`.
  Building with `-DPARCEL_KS_ORTHO_DEFAULT=mgs` makes MGS2 the default when `set_qr_algo()` is not called.
  Builds for PS-A (spack `ks_ortho=mgs`, `ps_a` in the tests) use MGS2 by default. The eigenvector output `EigenVec` accepts a length of n × nev or more, and
  writes only min(k_keep, length / n) vectors.

### Preconditioners

Selected with `set_precon_algo()` (`parcel_options.precon_algo` in the C API).

| Value | Constant | Description |
|---:|---|---|
| 0 | `PRECON_NONE` | None (default) |
| 1 | `PRECON_POINTJACOBI` | Point Jacobi (diagonal scaling) |
| 2 | `PRECON_BLOCKJACOBI` | Block Jacobi (ILU(0) per block). The number of blocks is set with `set_blockjacobi()`; by default it is determined automatically from the device (the number of threads per rank on CPUs, 2 rows per block on GPUs) |
| 4 | `PRECON_JACOBI` | Jacobi iteration (number of iterations with `set_jacobi()`, default 1) |

Scaling of the matrix and vectors is specified with `set_scaling_matrix()` / `set_scaling_vector()`
(if not specified, it is enabled automatically for block Jacobi and for low-precision configurations).

## Precision

The data precision of the matrix and vectors and the precision of the operations (SpMV, preconditioner, vector operations, inner products) can be chosen separately.
The types are `parcel_double` (double precision), `parcel_float` (single precision), `parcel_fp16` (half precision), and
`parcel_doubledouble` (double-double, about 32 digits).

- The arguments of the solver class template are the operation precisions, and the template argument of `solver()` is the data precision.
  If both are omitted, double precision is used (`parcel::krylov::pcg<> s; s.solver(A, x, b);`).
  ```cpp
  // Precisions of SpMV, preconditioner, vector operations, inner products
  parcel::krylov::pcg<parcel_float, parcel_float, parcel_double, parcel_double> s;
  ```
- In the verification samples and the C / Fortran wrappers, they are selected with the compile-time macros `parcel_solver_data_type` /
  `parcel_solver_calc_type` / `parcel_solver_calc_precon_type` / `parcel_solver_data_precon_mat_type` /
  `parcel_solver_data_precon_vec_type` (example: `-Dparcel_solver_calc_type=parcel_doubledouble`).
- The relative residual reachable at low precision is limited (as a guide: 1e-6 in single precision, 1e-3 to 1e-2 in half precision; a half-precision preconditioner
  can break down around 1e-6).
- With half-precision data or operation precision, do not use the CholeskyQR family of `pcagmres` (`CAGMRES_Q_IMPLICIT` and the moment method)
  (it gives NaN or does not converge regardless of s, t). The version that forms Q explicitly (the default `CAGMRES_Q_AUTO` with a QR other than CholeskyQR) can be used with the default (s, t).
- At low precision, the residual of `pgmres` can stagnate near the tolerance, and shortening the restart cycle can make it stop earlier
  (there is a case with half-precision operations where m = 15 converged earlier than m = 30).
- Even with double-double, the eigenvalues of the eigenvalue solvers are limited to double precision (because the scalars on the host side are double precision).

## Requirements

- A SYCL 2020 implementation (tested with AdaptiveCpp 24.10; the OpenMP backend on CPUs, CUDA / HIP on GPUs)
- MPI (for MPI-parallel use. A single-process version without MPI can also be built with `-DPARCEL_MPI=0`; see "Building without MPI" below)
- LAPACKE (`lapacke.h` and its implementation, e.g. OpenBLAS). It is used by the eigenvalue solvers, so it is required even when only solving linear equations
- C / Fortran compilers when using the C / Fortran API (tested with gcc / gfortran)
- Python 3.8 or later and numpy when using the Python package ("Python" below)

These can also be installed together with spack ("Installing with spack" below).

## Build

Since the library is header-only, the library itself does not need to be built. In your program,
include `Include/parcel_sycl.hpp`, compile with a SYCL compiler, and link LAPACKE.
The following is an example with AdaptiveCpp + Open MPI + OpenBLAS (making `mpicxx` call `acpp` underneath).

```sh
export OMPI_CXX=acpp        # Make Open MPI's mpicxx call acpp (MPICH_CXX for MPICH)

# CPU (OpenMP backend)
mpicxx -O3 --acpp-targets=omp -march=native -D_BLOCKING_LOOP_DOT_=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

# GPU (NVIDIA V100 = sm_70. sm_80 for A100, sm_90 for H100)
#   Do not use ladder splitting (-DPARCEL_SPMV_LADDER=1) on NVIDIA GPUs (see below). On V100, -Xcuda-ptxas -maxrregcount=255 is recommended
mpicxx -O3 --acpp-targets=cuda:sm_70 -Xcuda-ptxas -maxrregcount=255 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

# GPU (AMD MI300A. Use ladder splitting)
mpicxx -O3 --acpp-targets=hip:gfx942 -DPARCEL_SPMV_LADDER=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas

mpirun -n 4 ./quick_start
```

Executables for CPUs (OpenMP backend) use LLVM's `libomp.so`, but AdaptiveCpp does not embed its location in the executable, whether `acpp` is called directly or
through `mpicxx` (it embeds only the location of AdaptiveCpp's own libraries).
Add LLVM's `lib/` to `LD_LIBRARY_PATH` at run time, or add an RPATH at link time. Without either, the OS libomp is
loaded, or it is not found and the program cannot start.

```sh
# llvm_root is the location of the LLVM that AdaptiveCpp was built with (for spack, llvm_root=$(spack location -i llvm))
mpicxx -O3 --acpp-targets=omp -march=native -D_BLOCKING_LOOP_DOT_=1 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas -Wl,-rpath,$llvm_root/lib
```

When installed with spack, this RPATH is included in `$PARCEL_LDFLAGS` ("Installing with spack" below).

On V100 (sm_70), adding `-Xcuda-ptxas -maxrregcount=255` is recommended. With ptxas's default register allocation,
the loads of the matrix and vectors in kernels such as the Jacobi preconditioner are split and serialized, which makes them slow. This flag groups the loads together
(no spills occur, and the computed results do not change). On other GPUs the effect depends on the GPU generation, so measure and decide if necessary.

Ladder splitting (`-DPARCEL_SPMV_LADDER=1`; `PARCEL_PRECON_LADDER` for the preconditioner follows it) splits the inner loops of SpMV and the preconditioner into 4-2-1.
On AMD MI300A it is faster (0.66 times per iteration of pcg + Jacobi preconditioner), so use it. On NVIDIA GPUs it is slower than the flat loop on V100
(1.18 times in the same comparison), and there was no difference on A100 and H100 (GH200), so do not use it. The defaults of the spack recipe (`kernel=auto`) and of Python's
`PARCEL_KERNEL_FLAGS` follow this as well.

On A64FX (`--acpp-targets=omp.library-only` with the Fujitsu compiler as the host compiler),
add `-D_BLOCKING_LOOP_=1 -DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=0 -ffp-contract=fast` (blocking of all kernels, ladder splitting of SpMV only,
and contraction of multiply-adds into FMA instructions). The clang mode of the Fujitsu compiler does not contract multiply-adds by default, and AdaptiveCpp appends `-O3` after `-Kfast`
in `ACPP_OMP_CXX_FLAGS`, so `-Kfast` does not take effect either. `-ffp-contract=fast` reaches the Fujitsu compiler if placed on the command line.
On one A64FX node (NX=256, pcg + Jacobi preconditioner), one iteration becomes 3 % faster (10 % for SpMV). The number of iterations is the same, and the last digits of the results change
(about 1e-11 relative). Inside the double-double functions, `Include/ddlib.hpp` keeps FMA disabled.
In this build, storing the values of the Jacobi preconditioner is the default (the default of `PARCEL_JACOBI_ONTHEFLY` is 0, no on-the-fly computation; see "Memory and Speed" below).
The recommendations for A64FX are summarized as follows (C2, D, E and F are enabled automatically in this build; the code generated for x86 and GPU builds does not change).

| | Description | How to specify |
|---|---|---|
| A | Contract multiply-adds into FMA instructions | `-ffp-contract=fast` at build time |
| B | Keep OpenMP threads from sleeping even while waiting (reduces the fixed cost of launching and waiting for kernels) | `OMP_WAIT_POLICY=ACTIVE` and `KMP_BLOCKTIME=infinite` at run time |
| C2 | axpy with coefficients passed by value and gemv (the axpy-decomposed version), rewritten with contiguous range partitioning, copies into local variables, and explicit `std::fma` | Default (`PARCEL_A64FX_AXPY_C`, disabled with `-DPARCEL_A64FX_AXPY_C=0`) |
| D | In `pcg` (pstep = 1), fuse the updates of x and p, and the update of r with the inner product for ‖r‖, each into a single kernel | Default (`PARCEL_CG_FUSED`, disabled with `-DPARCEL_CG_FUSED=0`) |
| E | In preparing the right-hand side of the Jacobi preconditioner (D⁻¹b), skip the division when there is no vector scaling (the divisor is 1). On A64FX the division accounted for about 30 % of the preconditioner | Default (`PARCEL_JACOBI_NODIV_UNIT`, disabled with `-DPARCEL_JACOBI_NODIV_UNIT=0`) |
| F | When the preconditioner matrix is single precision (`precon_dataF` and the like), process two rows alternately in the Jacobi preconditioner iterations to overlap the multiply-add latencies (not used for double-precision matrices) | Default (`PARCEL_JACOBI_ROW2`, disabled with `-DPARCEL_JACOBI_ROW2=0`) |
| | Store the values of the Jacobi preconditioner (no on-the-fly computation) | Default (`PARCEL_JACOBI_ONTHEFLY` 0) |

C2, D, E and F do not change the expression for each element or the order of summation, so the results are bit-identical to those before the rewrite (confirmed on A64FX with the residual history over 200 iterations for C2 and D, and until convergence (306 iterations) for E and F).
Effect of E and F (one A64FX node, NX=256, pcg + Jacobi preconditioner, one iteration): the preconditioner section becomes 0.77 times in double precision and 0.70 times with `precon_dataF`,
and one iteration becomes 0.90 times and 0.87 times. The preconditioner with `precon_dataF` takes 1.06 times that of double precision (1.17 times before E and F).
With B the threads keep spinning, so it is counterproductive in placements where cores are contended (multiple threads or other processes on one core). In the recommended placement
(1 rank = 1 CMG, 12 threads) there is no problem. The example test settings (`wis_ody` / `wis_odyspack` in `tests/site_config.py`) set B.
Effect (A64FX, pcg + Jacobi preconditioner, time per iteration with 200 iterations fixed, ratio to the build with A only): on one node (4 ranks × 12 threads),
0.59 times for NX=64, 0.78 times for NX=128, and 0.99 times for NX=256 (the fewer the rows, the more the fixed cost matters and the larger the effect). For NX=512, 0.96 times on 4 nodes
and 0.84 times on 8 nodes; on 8 nodes it is 4 % faster than PETSc (flat MPI on the same number of nodes).

For the C / Fortran API, compile the wrapper `capi/wrapper_parcel_sycl.cpp` with a SYCL compiler and
link it with your program (link with the SYCL compiler).

```sh
mpicxx -O3 --acpp-targets=omp -I/usr/include/openblas -c capi/wrapper_parcel_sycl.cpp -o wrapper_parcel_sycl.o

# C
mpicc -O2 -fPIE -Icapi -c examples/sample_c.c -o sample_c.o
mpicxx --acpp-targets=omp -o sample_c sample_c.o wrapper_parcel_sycl.o -lopenblas -lm

# Fortran (compile the module capi/parcel_sycl_mod.f90 first)
mpif90 -O2 -fPIE -c capi/parcel_sycl_mod.f90
mpif90 -O2 -fPIE -c examples/sample_f.f90
mpicxx --acpp-targets=omp -o sample_f sample_f.o parcel_sycl_mod.o wrapper_parcel_sycl.o \
  $(mpif90 --showme:link) -lgfortran -lopenblas
```

AdaptiveCpp links as PIE, so build gcc / gfortran objects with `-fPIE`.
The internal precision of the wrapper is determined by `parcel_solver_*_type` when the wrapper is compiled
(defaults in `capi/parcel_capi_default.hpp`). The C++, C and Fortran samples in examples can be built together with `bash examples/build_examples.sh`
(the compilers, target and LAPACK are changed with environment variables; see the top of the script).

### Checking the Build

`bash examples/run_quick_checks.sh` builds and runs the quick starts (C++, C, Fortran, Python, and with MPI also C++ with 2 processes),
and checks whether the maximum relative error against the exact solution of a 1D Laplacian (n=1000) is 1e-6 or less. At the end it prints
`PASS n / FAIL m` on one line, and the exit code is 0 if all pass. The build (`build_examples.sh quick`, which runs SYCL compilations
in parallel) happens only the first time; if `examples/build/` holds a record of the settings and sources of the previous build and they are the same as now, the build is skipped and only the run is performed
(the run takes less than 1 minute). If the settings (`SYCL_TARGET`, `CXXFLAGS`, `PARCEL_MPI`, compilers, and so on) or the sources (headers, API, samples) have
changed, it shows those items and rebuilds. `--rebuild` always rebuilds.
The environment variables are the same as for `build_examples.sh` (`PARCEL_MPI=0` for no MPI, `BUILD_JOBS` for the number of parallel jobs). The launch command is
changed with `MPIEXEC` (default `mpirun`) and `MPIEXEC_NP_FLAG` (`-n`). Python is SKIPped if `parcel` cannot be imported. See the top of the script for details.

**Number of threads and where it runs**: if `OMP_NUM_THREADS` is not set, the script uses all available logical CPUs on a compute node (inside a job),
and 4 on a login node (or the number available, if fewer than 4). For the MPI run this is divided by the number of processes so that the total stays the same
(at least 1; 2 processes on a login node get 2 threads each). The number of available logical CPUs is the smallest of the CPU affinity (`nproc`),
the CPU limit of the cgroup, and the job's allocation on this node (PBS `NCPUS`, Slurm `SLURM_CPUS_ON_NODE`).
Whether it is a compute node is decided by the presence of a job environment variable (PBS `PBS_JOBID`, Fujitsu PJM `PJM_JOBID`, Slurm `SLURM_JOB_ID`),
and can be forced with `QUICK_NODE=compute` / `QUICK_NODE=login`. The decision and the values used are printed first. If `OMP_NUM_THREADS` is set,
its value is used (in a job the scheduler may set it: PBS `ompthreads` (a queue default of 1 is possible), Fujitsu PJM `--omp thread=`).
With Open MPI, `--bind-to none` is added unless `MPIEXEC_ARGS` specifies a placement (`--bind-to`, `--map-by`) (with 2 or fewer processes Open MPI binds each
process to one core, which would pack several threads onto one core).

**Where to run it**: in CPU environments, a login node is fine (follow the rules of the site). In GPU environments and on A64FX (Wisteria-Odyssey),
run it on a compute node (a short batch job or an interactive job): login nodes have no GPU, and A64FX executables do not run on the x86 login nodes.
When the MPI run (2 processes) is done in a GPU environment, the script does not assign one GPU per process (as the regression tests do with
`gpu_bind_*.sh`), so both processes may use the same GPU.
Fujitsu `mpiexec` (`MPIEXEC=mpiexec` whose real file is under `/opt/FJSV…`) writes the standard output of the ranks to the job log, where the check lines
cannot be read, so `RC_ONLY=1` (pass if the exit status is 0) is the default there. Use `RC_ONLY=0` to turn it off.

```sh
export OMPI_CXX=acpp
bash examples/run_quick_checks.sh                    # Last line: === PASS 6 / FAIL 0 (SKIP 0)  elapsed <seconds> s
PARCEL_MPI=0 bash examples/run_quick_checks.sh       # Without MPI
bash examples/run_quick_checks.sh --rebuild          # Rebuild
NO_BUILD=1 bash examples/run_quick_checks.sh         # Only run examples/build/ without checking the record
```

### Building without MPI

Add `-DPARCEL_MPI=0` and build directly with the SYCL compiler instead of the MPI compiler wrapper. `mpi.h` is not read,
and the MPI library is not linked. It runs as a single process and is executed without `mpirun`. The computation is the same as running the MPI build
with one process (the results match bit for bit). The default (when `PARCEL_MPI` is not specified) is with MPI.
For how to check this with the regression tests, see "Configurations without MPI (`<plat>_nompi`)" in `tests/regression/README.md`.

```sh
acpp -O3 --acpp-targets=omp -march=native -DPARCEL_MPI=0 \
  -I/usr/include/openblas -o quick_start examples/quick_start.cpp -lopenblas
./quick_start

# Samples in examples (gcc / gfortran for C / Fortran)
PARCEL_MPI=0 bash examples/build_examples.sh
```

User code works written the same way as with MPI. The differences are as follows.

- C++: Create the communicator with `parcel::comm comm(&q);` (corresponding to `parcel::comm comm(MPI_COMM_WORLD, &q);` with MPI).
  `mpi_rank()` is 0 and `mpi_size()` is 1. `set_crsMat(&comm, nstart, nend, n, nnz, ...)` can be used as is, but
  since one process holds everything, set `nstart = 0` and `nend = n - 1` (otherwise `std::invalid_argument`).
  `set_crsMat(n, nnz, rowptr, col, val)`, which does not take comm, can also be used. Functions that use MPI types
  (`comm::mpi_comm()`, `parcel::blas(q, MPI_Comm)`, and so on) do not exist. Separate the MPI calls in your code
  (`MPI_Init` and so on) with `#if PARCEL_MPI` (see `examples/quick_start.cpp`).
- C: Compile both the wrapper and the program that includes `parcel_sycl_c.h` with `-DPARCEL_MPI=0`.
  The function names and argument lists are the same, and the communicator argument (`PARCEL_C_COMM`) becomes an `int` and is not used (pass 0).
  Set `nstart = index_base` and `n_local = n_global` (otherwise `PARCEL_ERR_DISTRIBUTION`). Exceptions inside the library
  return `PARCEL_ERR_RUNTIME` instead of calling `MPI_Abort`.
- Fortran: `capi/parcel_sycl_mod.f90` is the same (the communicator integer is not used). Compile `examples/sample_f.f90` unchanged
  with gfortran together with `examples/sample_f_mpi_stub.f90`, a single-process substitute for the `mpi` module.

### Main Compile-Time Macros

| Macro | Default | Description |
|---|---|---|
| `_BLOCKING_LOOP_DOT_` | 0 | 1 enables blocking of reductions (dot / max). Intended for CPUs (OpenMP backend); faster by avoiding the fixed cost of barriers |
| `_BLOCKING_LOOP_` | 0 | 1 enables blocking of all kernels (for A64FX) |
| `_BLOCKING_LOOP_SPMV_` | 0 | 1 also enables blocking of SpMV (for A64FX. By default only SpMV uses a flat loop) |
| `PARCEL_SPMV_LADDER` | 0 | 1 splits the inner loop of SpMV into a 4-2-1 ladder (for AMD GPUs. Do not use on NVIDIA GPUs. No effect on CPUs) |
| `PARCEL_PRECON_LADDER` | `PARCEL_SPMV_LADDER` | Ladder splitting of the iteration loop of the preconditioner (set to 0 on A64FX) |
| `PARCEL_GEMV_AS_AXPY` | value of `_BLOCKING_LOOP_` | 1 decomposes gemv into repeated axpy (for A64FX) |
| `PARCEL_A64FX_AXPY_C` | 1 | 1 uses the versions of axpy with coefficients passed by value and of the axpy-decomposed gemv rewritten for A64FX. 0 uses the form before the rewrite (for A64FX. Same results. Effective only in `_BLOCKING_LOOP_` builds) |
| `PARCEL_CG_FUSED` | value of `_BLOCKING_LOOP_` | 1 fuses the kernels of one iteration of `pcg` (updates of x and p, update of r and ‖r‖. For A64FX. Same results. Can be set to 1 only in `_BLOCKING_LOOP_` builds) |
| `PARCEL_JACOBI_NODIV_UNIT` | 1 | Selects the version that skips the division in preparing the right-hand side of the Jacobi preconditioner when the divisor (the maximum of the vector scaling) is 1 (for A64FX. Same results. Effective only in `_BLOCKING_LOOP_` builds) |
| `PARCEL_JACOBI_ROW2` | value of `_BLOCKING_LOOP_` (0 if `PARCEL_PRECON_LADDER` is 1) | 1 processes the Jacobi preconditioner iterations two rows at a time alternately when the preconditioner matrix is single precision (for A64FX. Same results. Not used for double-precision matrices or on-the-fly computation. Can be set to 1 only with `_BLOCKING_LOOP_=1` and `PARCEL_PRECON_LADDER=0`) |
| `PARCEL_INT32` | undefined | If defined, makes `PARCEL_INT` (global row and column numbers and the like) 32 bits (default 64 bits) |
| `PARCEL_LOCAL_INT64` | undefined | If defined, makes the in-rank index `PARCEL_LINT` 64 bits (default 32 bits) |
| `PARCEL_FP_CONTRACT_OFF` | undefined | Enables `#pragma clang fp contract(off)` (effective only together with `-ffp-contract=fast-honor-pragmas`; see the notes below) |
| `PARCEL_DD_NO_FMA` | undefined | Do not use `fma` in double-double multiplication (for environments without FMA instructions) |
| `PARCEL_NO_CRS_CHECK` | 0 | 1 removes the input checks of `set_crsMat()` |
| `PARCEL_REQUIRE_IN_ORDER` | 1 | 0 removes the check "whether the queue is in-order" |
| `PARCEL_ALLOW_FAST_MATH` | undefined | If defined, allows building with `-ffast-math` (double-double and NaN detection are not guaranteed) |
| `PARCEL_ENABLE_ALL_SOLVERS` / `PARCEL_ENABLE_<solver>` | 1 | 0 excludes the solver from the build (to shorten compile time. Example: `-DPARCEL_ENABLE_ALL_SOLVERS=0 -DPARCEL_ENABLE_PCG=1`) |
| `PARCEL_NBLOCK` | 12 | The last-resort default for the number of partitions of blocked kernels. At run time, the environment variable `PARCEL_NBLOCK` takes precedence, followed by the number of OpenMP threads |
| `RESIDUAL_REPLACE` | undefined | If defined, `pcg` / `pbicgstab` also compute and print the true residual at every iteration (for diagnostics) |
| `PARCEL_JACOBI_ONTHEFLY` | 1 (0 for A64FX) | 1 does not store the values of (D⁻¹)(L+U) of the Jacobi preconditioner (`PRECON_JACOBI`), and computes them from the values of A and D⁻¹ at every application of the preconditioner. 0 stores the values (uses memory of one element of the preconditioner matrix type per nonzero of A, and the preconditioner becomes faster on GPUs; see "Memory and Speed" below). Default 0 in A64FX builds where `_BLOCKING_LOOP_` (or `_BLOCKING_LOOP_SPMV_`) is 1 |
| `PARCEL_KS_ORTHO_DEFAULT` | undefined (cgs2) | Default orthogonalization of `ks_arnoldi` (when `set_qr_algo()` is not called): `cgs2` / `mgs`. Builds for PS-A use `mgs` (MGS2 can be faster on CPUs with a small local length per thread and large caches). The discontinued run-time environment variable `PARCEL_KS_ORTHO` is not read (if set, rank 0 prints a one-line warning) |
| `PARCEL_PCG_NO_XHIST` | 1 | Default number of iterations over which `pcg` batches the solution updates (pstep). 1 gives 1 (update every iteration), 0 gives 4. The run-time environment variable `PARCEL_PSTEP` takes precedence |

Run-time environment variables:

| Environment variable | Description |
|---|---|
| `PARCEL_NBLOCK` | Number of partitions of blocked kernels |
| `PARCEL_PSTEP` | Number of iterations over which `pcg` batches the solution updates (default 1, 4 when built with `-DPARCEL_PCG_NO_XHIST=0`). Keeps a history of pstep vectors of p. Setting it to 4 reduces the reads and writes of the solution to 1/4 and makes it slightly faster, at the cost of the memory of 3 vectors (the way the solution is summed changes, so the rounding changes. The number of iterations is the same) |
| `PARCEL_PRECON_VERBOSE` | Prints the automatically determined preconditioner settings |
| `PARCEL_TIMER_SYNC` | 1 synchronizes the device before and after timing the inner products and the axpy with coefficients passed by value (to obtain an accurate time breakdown. By default the time of the inner products includes the time of the preceding axpy and the like) |
| `PARCEL_MEM_REPORT` | If set, counts the peak device memory allocated by PARCEL (`util_sycl::device_mem_peak_bytes()`; printed by the verification sample) |

## Main Defaults

| Setting | Default |
|---|---|
| `set_tolerance()` (all solvers) | 1.0e-8 (relative residual ‖r‖/‖b‖ for the linear solvers) |
| `set_max_iteration()` | 10000 (`set_ks_max_restart()` for `ks_arnoldi`) |
| Restart cycle of `pgmres` `set_restart(m)` | 30 |
| `set_restart(s, t)` of `pcagmres` | s = 8, t = 4 (restart cycle s·t = 32) |
| How `pcagmres` forms Q `set_explicit_q()` | `CAGMRES_Q_AUTO` |
| `set_ks_subspace(m, k_keep, nev)` of `ks_arnoldi` | m = 30, k_keep = 10, nev = 4 |
| `set_ks_max_restart()` of `ks_arnoldi` | 200 |
| Orthogonalization of `ks_arnoldi` `set_qr_algo()` | `QR_CGS` (CGS2. `QR_MGS` with `PARCEL_KS_ORTHO_DEFAULT=mgs` at build time) |
| Preconditioner `set_precon_algo()` | `PRECON_NONE` |
| Internal linear solver of the eigenvalue solvers (`inverse_power`, shift-invert of `ks_arnoldi`) | pgmres for `inverse_power` (pcg with `set_linsolve_algo(LINSOLVE_PCG)`), Jacobi preconditioner (1 iteration), restart cycle 30, tolerance 1e-10, maximum number of iterations 5000 (2000 for `ks_arnoldi`) |
| `set_precon_spectrum()` | false (eigenvalues of A) |

`pgmres` / `pcagmres` iterate exactly up to the maximum number of iterations even if it is not a multiple of the restart cycle
(the last cycle is shortened). The defaults of `parcel_options` for C / Fortran are read from the library by
`parcel_options_init()`.

## Minimal Example

Key points of `examples/quick_start.cpp` (solving a 1D Laplacian with pcg):

```cpp
#include "Include/parcel_sycl.hpp"

MPI_Init(&argc, &argv);
sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};  // An in-order queue is required
parcel::comm comm(MPI_COMM_WORLD, &q);

// Each rank owns contiguous rows [nstart, nend]. Row pointers are local (0-based), column numbers are global (0-based)
parcel::matrix<parcel_double> A(&q);
A.set_crsMat<double>(&comm, nstart, nend, n, nnz, rowptr, col, val);   // rowptr, col are PARCEL_INT*

parcel::vector<parcel_double> x(n, &q), b(n, &q);
x.setValue(x_host);  b.setValue(b_host);

parcel::krylov::pcg<> cg;               // Double precision if precision is omitted
cg.set_tolerance(1.0e-9);
cg.set_precon_algo(parcel::PRECON_JACOBI);
cg.solver(A, x, b);
x.getValue(x_host);  q.wait();
if (!cg.is_converged()) { /* reason via cg.get_converged_reason() */ }
```

The eigenvalue solvers are called in the form `s.solver(A, eigval, eigvec)` (`examples/example_pcg_ks.cpp`).
For C / Fortran, see `examples/sample_c.c` / `examples/sample_f.f90`.

## Notes

- **`set_crsMat()` modifies the argument arrays.** Column numbers are converted from global to local numbers, and the values in each row are reordered so that
  the columns of the own rank come first (row pointers do not change). Creating a second matrix (e.g. A − σI) from the same arrays
  converts them twice and breaks them, so pass separate arrays for each matrix or make a copy before calling.
- **Create the SYCL queue as in-order** (`sycl::property::queue::in_order{}`). This is because PARCEL submits kernels in sequence without
  chaining them with events. Passing a queue that is not in-order throws `std::invalid_argument`.
- **Input checks of the row-block partitioning and CRS.** `set_crsMat()` checks that the row blocks of the ranks cover all rows [0, n) without gaps or overlaps
  (they need not be in rank order), that `n == nend-nstart+1` with at least one row, that the row pointers are
  0-based, monotonically non-decreasing and end with nnz, and that the column numbers are within range; if any is violated, all ranks throw the same
  `std::invalid_argument`. To remove the checks, use `-DPARCEL_NO_CRS_CHECK=1`.
- **Integer types.** Global quantities (the total number of rows, `nstart` / `nend`, and the row pointers and global column numbers passed to `set_crsMat()`) use
  `PARCEL_INT` (default `int64_t`, `int32_t` with `-DPARCEL_INT32`), and the CRS indices on the device use `PARCEL_LINT`
  (default `int32_t`, `int64_t` with `-DPARCEL_LOCAL_INT64`). If the number of nonzeros of one rank (or the number of local rows + halo entries)
  exceeds 2³¹−1, `set_crsMat()` throws `std::overflow_error` on all ranks, so increase the number of ranks or
  build with `-DPARCEL_LOCAL_INT64`. The integers of the C / Fortran API are 64 bits regardless of `PARCEL_INT`
  (`parcel_int` = `int64_t` in C, `integer(parcel_int_kind)` in Fortran).
- **To make results match across rebuilds, add `-ffp-contract=off`.** By default AdaptiveCpp contracts multiply-adds into FMA
  (`-ffp-contract=fast`; on A64FX it is specified explicitly by the recommended flags), and which operations are contracted can change from build to build (especially with HIP). With the same executable
  the results are the same every time, but rebuilding changes the rounding, and in low-precision configurations the number of iterations can change.
- **Do not use `-ffast-math`.** It eliminates the error terms of double-double and the detection of NaN / Inf, so the library stops builds with this flag
  with `#error`.
- If device memory allocation fails, `util_sycl::device_bad_alloc` (derived from `std::bad_alloc`) is thrown.
  The failure can occur on only some of the ranks, so do not catch it and continue; stop the whole run.
- The degree of parallelism of applying the block Jacobi preconditioner equals the number of blocks, so it is slow on GPUs. On GPUs the Jacobi preconditioner is faster in most cases.
- The Jacobi preconditioner (`PRECON_JACOBI`) shares the row pointers and column numbers of matrix A without copying them, and by default (`PARCEL_JACOBI_ONTHEFLY=1`; except builds for A64FX)
  also reads the values of A at every application. The preconditioner is created inside `solver()` and destroyed when returning from `solver()`, so do not
  destroy or modify A while `solver()` is running (it is rebuilt at every call of `solver()`, so changing the values of A between calls is fine).
  If you create and use a preconditioner object (`parcel::precon_matrix`) yourself, do not destroy or modify the original matrix until you have finished using it.

## Memory and Speed

The default is a setting that uses less memory; the following two settings use more memory to run faster (the algorithm is the same for both).

| Setting | Additional memory (per rank) | Effect |
|---|---|---|
| `-DPARCEL_JACOBI_ONTHEFLY=0` (build time) | Values of the Jacobi preconditioner (`PRECON_JACOBI`): number of nonzeros of A × preconditioner matrix type (8 bytes for double precision) | No multiplications or diagonal checks in the preconditioner. The preconditioner becomes faster on GPUs (almost no change on CPUs) |
| `PARCEL_PSTEP=4` (run time) or `-DPARCEL_PCG_NO_XHIST=0` (build time) | History of p in `pcg`: 3 vectors of the local length (including the halo) | Reads and writes of the solution become 1/4. `pcg` becomes a few % faster |

For a 7-point stencil matrix (7 nonzeros per row), the former corresponds to about 7 vectors and the latter to 3 vectors.
In builds for A64FX (`_BLOCKING_LOOP_` or `_BLOCKING_LOOP_SPMV_` is 1), the former is enabled by default (the default of `PARCEL_JACOBI_ONTHEFLY` is 0), and
in double precision it uses 8 more bytes per nonzero of A. This is because on A64FX the on-the-fly computation shows up in the preconditioner time, and on one node (NX=256, pcg + Jacobi preconditioner)
storing the values makes the preconditioner 0.77 times and the total time to convergence 0.81 times. To reduce memory, specify `-DPARCEL_JACOBI_ONTHEFLY=1` explicitly.
Mixed precision with a single-precision preconditioner matrix (`-Dparcel_solver_data_precon_mat_type=parcel_float`) reads the double-precision values of A at every application
with the default on-the-fly computation, so it is hard to make it faster. Combined with `-DPARCEL_JACOBI_ONTHEFLY=0`, it reads single-precision values
(the additional memory is 4 bytes per nonzero of A). The mixed precision in the performance tests (`tests/perf/gen_perf_prec.py`) is measured with this combination.
With `-ffp-contract=off`, the former does not change the results bit for bit (in builds that allow fused multiply-add, the rounding may change).
The latter changes the way the solution is summed, so the results change at the level of rounding (the number of iterations is the same).

## Installing with spack

`spack/` contains spack recipes (`parcel-sycl`: headers and the C / Fortran API, `py-parcel-sycl`: the Python package) and
`spack/stack/install_stack.sh`, which installs everything from scratch, from obtaining spack through LLVM, AdaptiveCpp, OpenBLAS and Python to PARCEL.
MPI, CUDA / ROCm and the compilers are the existing ones on each computer (written in `spack/stack/envs/<env>/spack.yaml`).

```sh
bash spack/stack/install_stack.sh s86 $HOME/parcel_stack 16      # <env> <install prefix> [parallelism]
bash spack/stack/check_stack.sh $HOME/parcel_stack $HOME/parcel_check
. $HOME/parcel_stack/setup.sh
spack test run parcel-sycl py-parcel-sycl      # Run the quick start with what was installed (GPU versions on a node with a GPU)
spack load parcel-sycl+mpi+cuda py-parcel-sycl+mpi+cuda
mpicxx $PARCEL_CXXFLAGS -O3 -o prog prog.cpp $PARCEL_LDFLAGS      # Your program (with the same target and optimization)
```

For the bundled environment definitions (s86, Miyabi-C / G, PS-A / B, Wisteria-Aquarius / Odyssey) and notes, see [spack/README.md](spack/README.md).

## Python

`python/` is the Python package `parcel` (double precision, without / with MPI). The native layer (`python/src/parcel_py.cpp`)
calls PARCEL's C++ library directly. Install it with spack `py-parcel-sycl` or with `pip install .` in `python/`.

```python
import parcel
A = parcel.DistributedCSR.from_global(A_global)          # The rows assigned to this rank, from a scipy CSR matrix or the like
x, info = parcel.solve(A, b[A.local_slice()], method="pcg", precond="jacobi", tol=1e-10)
vals, vecs, info = parcel.eigs(A, method="ks_arnoldi", nev=4)
```

For details, see [python/README.md](python/README.md).

## Tests

`tests/` contains a full set of regression tests and performance tests. Both are **generators** of job scripts, and
the per-environment settings (queues, modules, compile flags, accounts to be charged) are collected in `tests/site_config.py`.
The bundled settings are examples for the environments used during development (x86 CPUs, A64FX, V100 / A100 / H100 / MI300A, macOS, and so on), so
rewrite them for your environment before use (`YOUR_PROJECT` / `YOUR_GROUP` / `YOUR_USER` are placeholders).

- **Regression tests** (`tests/regression/`): build the verification sample `tests/verify/parcel_krylov_sample_verify.cpp` in
  13 precision configurations, solve 7 solvers × matrices (Poisson / advection-diffusion / advection-diffusion with complex eigenvalues) × 4 preconditioners ×
  MPI 1/2/4 with NX=8 (n=512), and compare with reference solutions from a dense direct method and dense eigenvalue computation.
  A check that double-double has not degraded to double (`ddcheck`) is also included.
  ```sh
  python3 tests/regression/gen_regression.py s86_cpu     # Generate tests/regression/sub_*.sh
  bash tests/regression/sub_build_s86_cpu.sh             # Build the executables in bin/s86_cpu/
  for f in tests/regression/sub_regression_s86_cpu_*.sh; do qsub $f; done   # Submit from the package root
  bash tests/regression/summarize.sh s86_cpu             # Count the results in regression_results/
  ```
  There are also tests of the C / Fortran / Python APIs (`examples/sample_c.c`, `sample_f.f90`, `tests/regression/sample_py.py`) and
  examples of running in a spack environment installed from scratch (`s86spack_cpu` / `s86spack_gpu`).
  Every test checks the version that was run ("Version, Citation and Contact" below) against the value the generator takes from `Include/parcel_version.hpp`.
- **Version check** (`tests/check_version.py`, no build required): checks that the files that copy the version (the Python package, the spack recipes,
  `CITATION.cff`) agree with `Include/parcel_version.hpp`.
- **Performance tests** (`tests/perf/`): measure the time to solve NX=512 (n = 130 million) to convergence (relative residual 1e-8) with 4 / 8 / 16 devices
  (advection-diffusion for pgmres and pcagmres, Poisson for the others; the solver parameters are the library defaults). They use
  `bin/<env>/verify_double` built by the regression-test build. There is also `gen_perf_prec.py`, which compares mixed precision (`precon_dataF`) with double precision
  (`precon_dataF` is built without on-the-fly computation of the Jacobi preconditioner, `-DPARCEL_JACOBI_ONTHEFLY=0`; see "Memory and Speed" above).

For details, see [tests/regression/README.md](tests/regression/README.md) and [tests/perf/README.md](tests/perf/README.md).

## Directory Layout

```
README.md, CHANGELOG.md        This document and the changes in v1.2 (Japanese versions: README.ja.md, CHANGELOG.ja.md)
CITATION.cff, SUPPORT.md       Citation information and contact guide (Japanese version: SUPPORT.ja.md)
COPYING, COPYING.LESSER        Licenses (GPL-3.0 / LGPL-3.0)
doc/                           HTML of the manual (User Guide, API Reference): index_en.html (English), index.html (Japanese)
Include/                       The library itself (headers only). Include parcel_sycl.hpp
capi/                          C / Fortran API
  parcel_sycl_c.h                C declarations (parcel_solve_linear / parcel_solve_eigen and others)
  parcel_sycl_mod.f90            Fortran module
  wrapper_parcel_sycl.cpp        Implementation of the C / Fortran API (build with a SYCL compiler)
  parcel_capi_default.hpp        Compile-time defaults of the wrapper (internal precision, settings of the old API)
examples/                      Samples
  quick_start.cpp                Minimal C++ example (pcg. Checks the error against the exact solution)
  quick_start_c.c, quick_start_f.f90, quick_start.py
                                 Minimal examples solving the same problem with the C / Fortran API and the Python package (without / with MPI)
  example_pcg_ks.cpp             C++ example of linear (pcg) and eigenvalue (ks_arnoldi) solvers
  sample_c.c, sample_f.f90       Examples that call and verify all solvers with the C / Fortran API
  sample_f_mpi_stub.f90          Single-process mpi module for building sample_f.f90 and quick_start_f.f90 without MPI
  problem.f90                    Fortran routine that builds a 3D Poisson matrix in CRS (for the old API)
  build_examples.sh              Script that builds the above samples together
  run_quick_checks.sh            Build check: builds and runs the quick starts and reports pass / fail
python/                        Python package parcel (README.md, pyproject.toml, CMakeLists.txt)
  parcel/                        Python layer
  src/parcel_py.cpp, .h          Native layer (libparcel_py.so)
  envs/mac_brew.sh               Settings for building on macOS + Homebrew
spack/                         spack recipes and the mechanism for installing from scratch (README.md)
  spack_repo/parcel/             Recipes parcel-sycl / py-parcel-sycl
  stack/                         install_stack.sh, check_stack.sh, per-environment definitions envs/<env>/
tests/
  site_config.py                 Per-environment settings of the tests (rewrite before use)
  check_version.py               Version check (Include/parcel_version.hpp and the files that copy the version)
  verify/                        Verification sample and generation of test matrices (Poisson, advection-diffusion, unstructured)
  regression/                    Regression test generator, submission and summary scripts, ddcheck, Python API tests
  perf/                          Performance test generator and summary scripts
tools/
  make_tarball.py                Creates the release tarball and sha256 (the same tarball from the same contents)
```

## Version, Citation and Contact

The version is written in only one place, `Include/parcel_version.hpp` (this version is 1.2.0, released on 2026-10-23).

| Language | How to obtain the version |
|---|---|
| C++ | Macros `PARCEL_VERSION_MAJOR` / `_MINOR` / `_PATCH`, `PARCEL_VERSION_STRING` (`"1.2.0"`), `PARCEL_VERSION_NUMBER` (10200), `PARCEL_VERSION_GE(1, 2, 0)`. Functions `parcel::version_string()`, `parcel::get_version(&major, &minor, &patch)` |
| C | `parcel_get_version(&major, &minor, &patch)`, `parcel_version_string()` (the values of the header when the wrapper was built) |
| Fortran | `ierr = parcel_get_version(major, minor, patch)` (`integer(c_int)`), `call parcel_version_string(str)` |
| Python | `parcel.__version__` (the package version), `parcel.library_version()` (the version of the header the native layer was built with). Both have the same value |

The versions in the Python package (`python/pyproject.toml`), the spack recipes (`version("1.2.0")`) and environment definitions (`parcel-sycl@1.2.0`), and `CITATION.cff` are
copies, so when raising the version, rewrite them all consistently and check with `python3 tests/check_version.py`.

Citation information is in [CITATION.cff](CITATION.cff) (paper to cite: Y. Idomura et al., ScalA 17 (2017)),
and contact information is in [SUPPORT.md](SUPPORT.md). Contact: GitHub Issues <https://github.com/parcel-jaea/PARCEL-release/issues>

## Release Tarball

The release tarball is created with `tools/make_tarball.py`. The same contents produce the same tarball (the same sha256)
(the file order, timestamps, owners and compression header are normalized; test outputs and `__pycache__` are not included).

```sh
python3 tools/make_tarball.py --outdir <output directory>     # parcel_sycl-1.2.0.tar.gz and parcel_sycl-1.2.0.tar.gz.sha256
sha256sum -c parcel_sycl-1.2.0.tar.gz.sha256        # Verification by the recipient (shasum -a 256 -c on macOS)
python3 tools/make_tarball.py --check              # Whether the sha256 written in the spack recipe matches the tarball of the current package
```

The spack recipe (`version("1.2.0", sha256=...)`) is also included in the tarball, so when it goes into the tarball it is always in the form
without the sha256 (`version("1.2.0")`). Writing the sha256 into the local recipe with `--pin` does not change the tarball.
The tarball is placed in GitHub Releases (<https://github.com/parcel-jaea/PARCEL-release/releases>, tag `v1.2.0`), and the `url` of the recipe points there.
For the steps to install from the tarball with spack, see "Installing from the Release Tarball" in [spack/README.md](spack/README.md).

## License

GNU Lesser General Public License version 3 (LGPL-3.0-only). See [COPYING.LESSER](COPYING.LESSER) and
[COPYING](COPYING). Each source file begins with `SPDX-License-Identifier: LGPL-3.0-only` and a copyright line.

Copyright (c) 日本原子力研究開発機構 システム計算科学センター HPC・DX基盤技術開発室
(Japan Atomic Energy Agency, Center for Computational Science & e-Systems)

Distribution site: <https://ccse.jaea.go.jp/software/PARCEL/>, public repository (Releases of the release tarball): <https://github.com/parcel-jaea/PARCEL-release>
