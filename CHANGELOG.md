English | [日本語](CHANGELOG.ja.md)

# CHANGELOG

## v1.2 (2026-10-23)

Changes from PARCEL_SYCL v1.1.

### Added

- **Eigenvalue solver `parcel::krylov::ks_arnoldi`** (Arnoldi + Krylov-Schur restart).
  Computes the `nev` eigenpairs of largest absolute value, judging convergence by the Ritz residuals. The restart uses the real Schur form, so
  complex-conjugate pairs of eigenvalues of nonsymmetric matrices are also handled. With `set_shift(σ)` in shift-invert mode, the internal pgmres
  solves (A − σI) z = v and the eigenvalues near σ are computed. The orthogonalization is chosen with `set_qr_algo()` (the same constants as `pgmres`;
  in `ks_arnoldi` both are applied twice): `QR_CGS` is block classical Gram-Schmidt twice (CGS2, default) and `QR_MGS` is
  modified Gram-Schmidt twice (MGS2). Any other value gives `std::invalid_argument`. In C / Fortran it is `ks_qr_algo` of `parcel_options`,
  in Python `orthogonalization` of `eigs()` (`"cgs"` / `"mgs"`). On CPU configurations with a small local length per rank
  MGS2 can be faster, and on CPUs with a large local length and on GPUs CGS2 is faster. The accuracy of the eigenvalues is the same.
  The environment variable `PARCEL_KS_ORTHO` is not read (if it is set, rank 0 writes a one-line warning to standard error at the first `solver()`).
- **Eigenvalue solver `parcel::krylov::inverse_power`** (inverse power method). Computes the eigenvalue of smallest absolute value.
  The internal linear solver is pgmres (default) or pcg.
- **Eigenvalues of the preconditioned matrix A K⁻¹**: `set_precon_spectrum(true)`, common to the eigenvalue solvers. The default (false) gives the eigenvalues of A.
- **Settings of the internal linear solver of the eigenvalue solvers**: `set_linsolve_algo / _tolerance / _max_iteration / _restart /
  _precon_algo / _blockjacobi / _jacobi / _scaling_matrix / _scaling_vector`.
- **Convergence status API** (all solvers): `get_converged_reason()` (`CONVERGED_RTOL` / `DIVERGED_ITS` /
  `DIVERGED_NANORINF` / `DIVERGED_BREAKDOWN`), `is_converged()`, `get_iterations()`, `get_residual()`,
  `get_true_residual()`. The eigenvalue solvers have `get_linsolve_failures()`, and `ks_arnoldi` has `get_ks_nconv()`.
  Rank 0 writes one line ` converged_reason = ... , its = ...` at the end of `solver()`.
- **API for calling all 7 solvers from C / Fortran** (`capi/parcel_sycl_c.h`, `capi/parcel_sycl_mod.f90`,
  `capi/wrapper_parcel_sycl.cpp`): `parcel_solve_linear` / `parcel_solve_eigen` (`_f` for Fortran),
  the settings structure `parcel_options` (`parcel_options_init()` reads the defaults from the library), the result structure
  `parcel_result`, uneven row partitioning, and error codes (`PARCEL_ERR_*`). The v1.1 `run_parcel_sycl_c` / `_f` are
  kept for compatibility.
- **Choice of how `pcagmres` builds Q**: `set_explicit_q(CAGMRES_Q_AUTO / CAGMRES_Q_EXPLICIT / CAGMRES_Q_IMPLICIT)`.
  `CAGMRES_Q_IMPLICIT` is the Cholesky QR version that does not form Q explicitly (the algorithm of the v1.1 `cholQR_pcagmres`).
  If no QR is specified it uses `QR_CHOLESKYQR`; for a symmetric matrix, no preconditioning and even s, the moment method
  `QR_CAGMRES_CHOLESKYQR` can also be chosen.
- **Input checks**:
  - `set_crsMat()` checks the row-block partitioning (all ranks together cover [0, n) with no gaps or overlaps) and the local CRS (the ranges of the row pointers and column indices),
    and if they are broken all ranks throw the same `std::invalid_argument` (can be disabled with `-DPARCEL_NO_CRS_CHECK=1`).
  - Out-of-range values of arguments to the setting functions and to `solver()` give `std::invalid_argument`.
  - If the SYCL queue is not in-order, `std::invalid_argument` (can be disabled with `-DPARCEL_REQUIRE_IN_ORDER=0`).
  - If a device memory allocation fails, `util_sycl::device_bad_alloc` (derived from `std::bad_alloc`) is thrown.
  - A build with `-ffast-math` is stopped by `#error` (because it breaks double-double and NaN detection;
    allowed with `-DPARCEL_ALLOW_FAST_MATH`).
- Runtime environment variable `PARCEL_TIMER_SYNC=1`: synchronizes the device before and after the timing of inner products and of axpy with coefficients passed by value
  (by default there is no synchronization, and the inner-product time includes the time of the preceding kernel (such as the axpy of MGS). Use it to get an accurate breakdown of the time).
- Runtime environment variable `PARCEL_MEM_REPORT`: when set, PARCEL counts the current amount and the peak of the device memory it allocated with USM
  (readable with `util_sycl::device_mem_peak_bytes()`; the verification sample prints it as `PARCEL device peak` after the solver).
- The memory report of the verification sample (`tests/verify/`) (`host_hugetlb_mib` of `[MEM]` and `[MEMG]`) now includes the amount of large pages (HugeTLB) (HugetlbPages in
  `/proc/self/status`, the value at the time of the report). On A64FX (Fujitsu compiler) the heap is placed on large pages, which are not counted in the peak resident memory (VmHWM).
  Host memory should be read as VmHWM + HugetlbPages. On environments that do not use large pages it is 0.
- **Switches to limit which solvers are built**: `PARCEL_ENABLE_ALL_SOLVERS` / `PARCEL_ENABLE_<solver>` (all by default).
- `PARCEL_FP_CONTRACT_OFF`: enables `#pragma clang fp contract(off)` (effective only together with
  `-ffp-contract=fast-honor-pragmas`; to get identical results across rebuilds, `-ffp-contract=off` is recommended).
- **Build without MPI** `-DPARCEL_MPI=0`: does not read `mpi.h`, and builds a single-process version without the MPI compiler wrapper.
  Create it with `parcel::comm comm(&q)`; `set_crsMat(&comm, nstart, nend, ...)` and the others are used in the same form as with MPI.
  The C / Fortran APIs use the same functions, and the communicator argument is not used. The results are bitwise identical to a single process with MPI.
  For usage see the "Building without MPI" section of README.md.
- Added configurations without MPI (`<plat>_nompi`, `-DPARCEL_MPI=0`, a single process launched directly) to the regression tests. The targets are
  `REGRESSION_NOMPI` in `tests/site_config.py`. Added `tests/regression/compare_nompi.py` to compare against MPI with np=1.
- Added tests of the C / Fortran APIs (`examples/sample_c.c` / `sample_f.f90`) to the regression tests (`lang_c` / `lang_f`, 1 item = 1 case).
  The build is `tests/regression/sub_build_lang_<plat>.sh`, and the targets are `REGRESSION_LANG` in `tests/site_config.py`.
  `tests/regression/summarize.sh` also prints the counts per language.
- C / Fortran / Python versions of the quick start (`examples/quick_start_c.c`, `quick_start_f.f90`, `quick_start.py`): solve the same problem as `quick_start.cpp`
  (the 1D Laplacian with n = 1000 by pcg without preconditioning) with the C / Fortran APIs and the Python package. The same source works without and with MPI.
- **Checking the build**: the 4 quick starts (C++, C, Fortran, Python) compute the maximum relative error against the exact solution x[i] = (i + 1)(n − i) / 2
  over the rows owned by each rank, gather it, and judge it with the threshold 1e-6 (`check: max relative error = ... (limit 1.0e-06) PASS`; on failure the exit code is 1).
  `examples/run_quick_checks.sh` (new) builds and runs them, with MPI also runs the C++ one with 2 processes,
  and prints `PASS n / FAIL m` on one line (without MPI `PARCEL_MPI=0`, launch commands `MPIEXEC`, `MPIEXEC_NP_FLAG` and so on; Python is SKIP if `parcel`
  is not available). The build is skipped if `examples/build/build_stamp.txt` (settings, compiler locations, cksum of the sources) matches the current state; otherwise
  the differing items are shown and it is rebuilt (`--rebuild` always rebuilds). `build_examples.sh` also builds the C / Fortran quick starts and
  runs the compilations concurrently (`BUILD_JOBS`, default is the smaller of the number of targets and the number of cores). Added the target `quick` (only the 3 quick starts) and
  `--print-stamp`, which prints the record. Added post-install tests to the spack recipes (`spack test run parcel-sycl
  py-parcel-sycl`), and added the variant `scipy` (default on, adds SciPy as a dependency) to `py-parcel-sycl`.
  Default number of threads (the same in both): if `OMP_NUM_THREADS` is not set, all available logical CPUs on a compute node (a job environment variable
  `PBS_JOBID`, `PJM_JOBID` or `SLURM_JOB_ID` is present), and 4 on a login node (divided by the number of processes for the MPI run; forced with `QUICK_NODE=compute|login`).
  The decision and the values used are printed. With Fujitsu mpiexec, `RC_ONLY=1` is the default (`run_quick_checks.sh`: an `mpiexec` whose real file is under `/opt/FJSV…`;
  spack: `+fj +mpi`). With Open MPI, `--bind-to none` is added unless a placement is given. With Intel MPI, the spack tests build C and Fortran with the
  generic wrappers (`mpicc`, `mpif90`) and spack's compiler (gfortran, etc.) through `I_MPI_CC` and `I_MPI_F90` (`mpiifx` (ifx) cannot read modules built by gfortran).
- **Python package `parcel`** (`python/`): calls all 7 solvers from a native layer independent of the C / Fortran APIs (`python/src/parcel_py.cpp`)
  (`parcel.solve` / `parcel.eigs`, double precision, without / with MPI). Install with pip (scikit-build-core) or with spack's
  `py-parcel-sycl`. See `python/README.md`.
- **spack recipes and a mechanism for installing from scratch** (`spack/`): `parcel-sycl` (headers, C / Fortran APIs) and `py-parcel-sycl`;
  `spack/stack/install_stack.sh`, which installs everything from fetching spack through LLVM, AdaptiveCpp, OpenBLAS, Python and PARCEL, and the checker
  `check_stack.sh`; definitions of 7 environments (`spack/stack/envs/`; Wisteria-Odyssey uses `+fj` with the Fujitsu compiler, MPI and LAPACK). A template `spack/stack/envs/generic/` to be adapted for other Linux machines (checked on Ubuntu 24.04 and Rocky Linux 9 (both on WSL2)). On V100 (`cuda_arch=70`), `-Xcuda-ptxas -maxrregcount=255` is added automatically.
  See `spack/README.md`.
- Added a test of the Python API (`tests/regression/sample_py.py`; the same problem, items and judgment as `sample_c.c`) to the regression tests (`lang_py`).
  The build is `sub_build_lang_<plat>.sh` (builds the native layer as `bin/<plat>/libparcel_py.so`), and the targets are
  `REGRESSION_LANG_PY` in `tests/site_config.py`. For the C / Fortran / Python tests, `parcel_from` selects whether to build from the package sources (default) or to use a PARCEL
  installed with spack or the like.
- Example settings for running the regression and performance tests in a spack environment installed from scratch (`s86spack_cpu` / `s86spack_gpu` in `tests/site_config.py`).
  For the regression tests there are also examples for Miyabi-C / Miyabi-G / PS-A / PS-B / Wisteria-Aquarius / Wisteria-Odyssey (`miy_cspack` / `miy_gspack` / `ps_aspack` /
  `ps_bspack` / `wis_aquspack` / `wis_odyspack`). Odyssey is built with the Fujitsu compiler and MPI, and the standard output of the ranks, which Fujitsu's `mpiexec`
  writes to the job log, is captured to files with `-std` and used for judgment (`rank_out` in `tests/site_config.py`).
  Every example tests the PARCEL installed in spack, including the verification sample (C++) (with `parcel_from="installed"` the verification sample and
  ddcheck are built with the installed headers, and C / Fortran / Python use `parcel-sycl` / `py-parcel-sycl`).
- Library-defaults mode of the verification sample (`PARCEL_LIB_DEFAULTS=1`): does not call the solver parameter setting functions with the sample's values and solves with the library defaults
  (values given explicitly by environment variables are still used). The values actually used are printed on the `[LIB_DEFAULTS]` line. Used by the performance tests.
- With / without MPI and the language (verification sample (C++) / C / Fortran / Python) of the regression tests can now be selected independently at every stage: generation (`gen_regression.py --plat --mpi --lang`),
  build (`sub_build_<plat>.sh`, `sub_build_lang_<plat>.sh [c] [fortran] [python]`), submission (`--mpi` / `--lang` of `feed_regression.sh`,
  kinds `cpp` / `lang_c` / `lang_f` / `lang_py`) and summary (`summarize.sh --mpi --lang`).
  A selective generation regenerates only the outputs of the selected combinations.
- Regression tests on machines without a scheduler (`sched="local"` in `tests/site_config.py`; `tests/regression/run_local.sh` builds,
  runs and summarizes). Added the environment `mac` (macOS + Homebrew) as an example.
- Examples `examples/quick_start.cpp` (minimal pcg) and `examples/example_pcg_ks.cpp` (pcg + ks_arnoldi).
- Generators for the regression tests (`tests/regression/`) and the performance tests (`tests/perf/`), the verification sample
  (`tests/verify/parcel_krylov_sample_verify.cpp`; compares against reference solutions from a direct method and a dense eigenvalue computation),
  and the double-double degradation detection (`tests/regression/ddcheck/`).
  The individual thresholds of the regression tests for low precision (pcg and pbicgstab with `calcF16` / `dataF16`) are common to CPUs and GPUs.
  The maximum number of iterations in the regression tests is 10000 for the linear solvers, 5000 for power (20000 with the block Jacobi preconditioner), and 200 for inverse_power.
  The number of threads per rank in CPU environments is 5 on s86, 12 (1 CMG) on A64FX, 7 on Miyabi-C and 7 on PS-A (Miyabi-C and PS-A place several ranks in one NUMA domain).
  The number of blocks of the block Jacobi preconditioner on CPUs is determined by the number of threads per rank.
  The performance tests measure the time to solve NX=512 to convergence (relative residual 1e-8, at most 10,000) on 4 / 8 / 16 devices (a socket for CPUs, one card for GPUs),
  3 times each (the fastest run is reported). The solvers for nonsymmetric matrices (pgmres m=60, pcagmres explicit / implicit s = t = 8) use convection-diffusion
  (c = 0.05), and pcg, pbicgstab and the eigenvalue solvers use Poisson. The solver parameters are the library defaults (only the internal solver of inverse_power
  is set to pcg with the Jacobi preconditioner; with the default pgmres, restart 30, the internal solve of Poisson reaches the limit and is 10 to 14 times slower). Mixed precision (`precon_dataF`)
  compares pcg and pgmres with the Jacobi preconditioner against double precision (`tests/perf/gen_perf_prec.py`). The mixed-precision executables are built by
  `tests/perf/sub_build_prec_<plat>.sh` in `bin/<plat>/perf_prec/`, and `precon_dataF` uses the Jacobi preconditioner without on-the-fly computation
  (`-DPARCEL_JACOBI_ONTHEFLY=0`) (with the default on-the-fly computation, every preconditioning reads the double-precision values of A, and the benefit of single precision hardly shows).
  The double precision used for comparison and `precon_dataF` of the regression tests keep the default.
  The orthogonalization of ks_arnoldi is measured with the default CGS2, except on PS-A (`ps_a`), which uses MGS2 (`PARCEL_KS_QR=mgs` of the verification sample
  calls `set_qr_algo(QR_MGS)`; `job_args` of `PERF` in `tests/site_config.py`). On CPUs with a small local length per thread and a large cache, MGS2 can be faster.
- Macro `PARCEL_KS_ORTHO_DEFAULT`, which selects at build time the default orthogonalization of `ks_arnoldi` (when `set_qr_algo()` is not called)
  (`cgs2` / `mgs`; CGS2 if undefined). Added the variant `ks_ortho` (default `cgs2`) to the spack recipe, and set it to `mgs` in the PS-A definition (`spack/stack/envs/psa`) and in
  the tests `ps_a` / `ps_aspack` (MGS2 is the default on PS-A). A build that does not define the macro has an unchanged preprocessed source.
- Example test environment settings (`tests/site_config.py`): an RPATH to the location of LLVM's `libomp.so` is added at link time (`LINKOPT`; AdaptiveCpp
  adds none, either with `acpp` or through the MPI wrapper, so without `LD_LIBRARY_PATH` in the job the OS libomp is loaded). The PBS placement
  (`place`) can now be specified, and on PS-B (1 vnode = 1 APU) jobs that fit in one node use `place=pack` and jobs that do not use
  `place=exclhost` (GPU assignment picks GPUs by their order within the node, so a scattered job uses the GPUs of other jobs).
  When np in the regression tests exceeds the number of devices per node (`node_chunks`), the number of ranks per node in the launch command is set to `node_chunks`,
  packing the ranks into ceil(np / node_chunks) nodes (spreading over nodes changes the order of reductions and the iteration counts of low-precision configurations).
- **Version information**: `Include/parcel_version.hpp` (read by `parcel_sycl.hpp`) with `PARCEL_VERSION_MAJOR` / `_MINOR` / `_PATCH`,
  `PARCEL_VERSION_STRING` (`"1.2.0"`), `PARCEL_VERSION_NUMBER`, `PARCEL_VERSION_GE(a, b, c)`, and `parcel::version_string()` /
  `parcel::get_version()`. `parcel_get_version()` / `parcel_version_string()` in the C / Fortran APIs, and `parcel.library_version()` in Python.
  The version of the Python package changed from 0.1.0 to 1.2.0, the same as the main package. The files that copy the version (the Python package, the spack recipes,
  `CITATION.cff`) are checked against each other by `tests/check_version.py`, and the regression tests compare the version that was run with the value the generator takes from the header
  (the `[VERSION]` line of the verification sample, and the `version` item of the C / Fortran / Python tests).
- `SPDX-License-Identifier: LGPL-3.0-only` and a copyright line at the top of every source file. `CITATION.cff` (citation information) and
  `SUPPORT.md` (how to get support).
- `tools/make_tarball.py`, which creates the release tarball (the same contents give the same tarball and sha256; the name is `parcel_sycl-1.2.0.tar.gz`).
  The spack recipe has the `url` of the release tarball (the GitHub Release `https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz`) and
  the sha256 written by `--pin` (the recipe included in the tarball is the form without sha256). The version in the spack recipes and environment definitions is `1.2.0`
  (`version("1.2.0")`, `parcel-sycl@1.2.0`; `tests/check_version.py` also checks the `@` versions in the environment definitions and in README.md).
- Contact: GitHub Issues of the public repository (`SUPPORT.md`, contact in `CITATION.cff`). `CITATION.cff` includes the public repository
  (`https://github.com/parcel-jaea/PARCEL-release`) and the paper to cite (Idomura et al., ScalA 17, 2017).
- **HTML manual included** (`doc/`): the User Guide and the API Reference in English and Japanese (entry `doc/index_en.html` for English,
  `doc/index.html` for Japanese). It opens directly in a web browser and loads nothing from external sites (math in MathML, search with an in-page index).
  The Developer Guide is not included.
- **Descriptive md files in English and Japanese**: `README.md`, `CHANGELOG.md`, `SUPPORT.md`, `spack/README.md`, `python/README.md`,
  `tests/regression/README.md` and `tests/perf/README.md` are in English, and the Japanese versions are `*.ja.md` in the same place (e.g. `README.ja.md`).
  Each file starts with links to the other language. `CITATION.cff` and `COPYING*` remain single files.

### Changed (Behavior and API)

- Removed ladder splitting (`-DPARCEL_SPMV_LADDER=1 -DPARCEL_PRECON_LADDER=1`) from the recommendation for NVIDIA GPUs (`+cuda` / `cuda:sm_*`; V100, A100, H100)
  (`-Xcuda-ptxas -maxrregcount=255` for V100 is kept). On V100 ladder splitting was slower than the flat loop, and on A100 and H100 (GH200) there was no difference.
  Changed `kernel=auto` of the spack recipe, the default of `PARCEL_KERNEL_FLAGS` in Python, the s86 GPU, Miyabi-G and Aquarius in `tests/site_config.py`
  (`ladder=False`), and the build examples in README.md. AMD GPUs (MI300A) keep ladder splitting (it is faster). CPUs are unchanged.

- The recommended kernel optimization for macOS + Homebrew is now blocking of the reductions only (`-D_BLOCKING_LOOP_DOT_=1`), the same as other CPUs
  (`python/envs/mac_brew.sh`; previously blocking of all kernels for A64FX, `-D_BLOCKING_LOOP_=1`). On Apple M4 the difference per iteration is within ±4 %, and
  memory is smaller because the values of the Jacobi preconditioner are not stored. `mac` of the regression tests already used this setting. With Homebrew's AdaptiveCpp,
  `--acpp-targets=omp` and `omp.library-only` give the same compilation.

- Changed the default of the macro `MAX_ITERATION` for the maximum number of iterations of the verification sample (`tests/verify/parcel_default.hpp`) and of the C / Fortran wrapper (`capi/parcel_capi_default.hpp`)
  from 10 to 10000 (the same as the library default of `set_max_iteration()`). This changes the default of `PARCEL_MAX_ITER` of the verification sample and the maximum number of iterations of the old API
  `run_parcel_sycl_c` / `_f`. The new API (`parcel_solve_linear` and so on) and the library itself are unchanged.

- **Defaults** (so that problems can be solved without specifying anything):

  | Setting | v1.1 | v1.2 |
  |---|---|---|
  | `set_tolerance()` | 1.0e-4 | 1.0e-8 |
  | `set_max_iteration()` | 100 | 10000 |
  | `set_restart()` of `pgmres` | 4 | 30 |
  | `set_restart(s, t)` of `pcagmres` | (1, 1) | (8, 4) |
  | `set_qr_algo()` of `pcagmres` | `QR_CGS` | `QR_MGS` |

  The QR of `pcagmres` was changed to MGS because CGS with (s, t) = (8, 4) can make the residual grow considerably in the first cycle
  (with `CAGMRES_Q_IMPLICIT`, if no QR is specified it is still `QR_CHOLESKYQR`). The default of `cagmres_qr_algo` in the C API is also
  `PARCEL_QR_MGS`, and with `CAGMRES_Q_IMPLICIT` this value is treated as "unspecified" (explicitly giving `PARCEL_QR_CGS` is `PARCEL_ERR_ARG`).
  For `ks_arnoldi`, added in v1.2, the maximum number of restarts is 200, and the preconditioner of the internal linear solver of the eigenvalue solvers is Jacobi (1 iteration).
  To get the same behavior as v1.1, give the old values explicitly with the setting functions above. `pgmres` holds (restart cycle + 1) work vectors,
  so the default memory usage increases.
- **Integer types**: the default of the type `PARCEL_INT` of global quantities (the total number of rows, `nstart` / `nend`, the row pointers and column indices passed to `set_crsMat()`)
  changed from `int32_t` to `int64_t` (32 bits with `-DPARCEL_INT32`; the v1.1 `-DPARCEL_INT` is
  accepted for compatibility and gives 64 bits). The CRS indices on the device use the new type `PARCEL_LINT` (default `int32_t`,
  64 bits with `-DPARCEL_LOCAL_INT64`), keeping the memory traffic of SpMV the same as with 32 bits.
  If the number of nonzeros of a rank (or the number of local rows + halo elements) does not fit in `PARCEL_LINT`, `set_crsMat()` throws
  `std::overflow_error` on all ranks.
- **Integers in the C / Fortran APIs are 64-bit** (C: `parcel_int` = `int64_t`, Fortran: `integer(parcel_int_kind)` =
  `integer(c_int64_t)`). The same applies to `n` / `nnz` / `crsCol` / `crsRow_ptr` of the old API `run_parcel_sycl_c` / `_f`
  (**ABI change**: rebuild the wrapper, the Fortran module and the user programs together).
- **The precision template arguments can be omitted**: `parcel::matrix<>` / `parcel::vector<>` and the data precision of `solver()` of each solver
  now default to `parcel_double` (you can write `parcel::krylov::pcg<> s; s.solver(A, x, b);`).
- **Template arguments of `pbicgstab`**: removed the unused leading `Tdata`, making them
  `<Tspmv, Tprecon, Taxpy, Tdot>`, the same as the other solvers. If you specified 5 in v1.1, drop the first one
  (in v1.1, specifying 4 shifted them by one, and SpMV, the preconditioner and the inner products were computed in unintended precisions).
- **The `cholQR_pcagmres` class was removed.** Specify `set_explicit_q(parcel::krylov::CAGMRES_Q_IMPLICIT)` for `pcagmres`
  (the QR is `QR_CHOLESKYQR` by default). Specifying `PARCEL_ENABLE_CHOLQR_PCAGMRES` stops with `#error`.
- **Added an argument at the end of `pgmres::solver()`**: `parcel::precon_matrix<...> *external_precon = nullptr`
  (it has a default value, so existing calls work as they are).
- **`pgmres` / `pcagmres` no longer round the maximum number of iterations down to a multiple of the restart cycle.** They iterate exactly up to the limit
  (shortening the last cycle). When the limit is a multiple of the cycle, or when they converge before the limit, the results are unchanged.
- **`power` / `inverse_power` stop at the iteration where NaN / Inf appears** and return `DIVERGED_NANORINF`
  (returning the last finite eigenvalue and the normalized vector).
- **The initial vector of `power`** changed from all ones to a combination of sin/cos built from the global row index (independent of the number of MPI processes).
  The all-ones vector is orthogonal to the eigenvector of the largest eigenvalue of the 3D Poisson problem (NX even), and that eigenvalue could not be reached.
- **Eigenvector output `EigenVec` of `ks_arnoldi`** is accepted if its length is at least n × nev, and only min(k_keep, length / n) vectors are written
  (if only the top nev vectors are needed, allocate n × nev instead of n × k_keep).
- The number of blocks of the block Jacobi preconditioner and the scaling defaults are determined automatically from the device and the precision
  (when `set_blockjacobi()` / `set_scaling_*()` are not called).
- **The defaults are now the low-memory settings** (the algorithm is the same; for settings that use more memory to be faster, see the "Memory and Speed" section of README.md):
  - The Jacobi preconditioner (`PRECON_JACOBI`) does not store the values of (D⁻¹)(L+U), and computes them from the values of A and D⁻¹ at every preconditioning
    (`PARCEL_JACOBI_ONTHEFLY`, default 1). Memory decreases by one element of the preconditioner matrix type per nonzero of A, but on GPUs
    the preconditioner becomes slower (almost unchanged on CPUs). The values are computed with the same expression as at construction, so with `-ffp-contract=off` the results are bitwise
    identical. When matrix scaling is used (enabled by default for half-precision preconditioning), or when any of the coefficient matrix, the preconditioner matrix or the computation type of the preconditioner is
    double-double, the values are stored as before. Preconditioners in combinations that store values use the same kernel as a `-DPARCEL_JACOBI_ONTHEFLY=0`
    build (which only reads the stored values), so they do not become slower (which kernel to use is decided only once, before the preconditioner is called).
    `-DPARCEL_JACOBI_ONTHEFLY=0` gives the previous behavior.
    For A64FX builds (`_BLOCKING_LOOP_` or `_BLOCKING_LOOP_SPMV_` is 1) the default is 0 (values stored). On one A64FX node
    (NX=256, pcg + Jacobi preconditioner), storing the values makes the preconditioner 0.77 times and the total time to convergence 0.81 times (the on-the-fly computation is not hidden).
    Memory increases by one element of the preconditioner matrix type (8 bytes in double precision) per nonzero of A. Giving `-DPARCEL_JACOBI_ONTHEFLY=1` explicitly selects the low-memory side.
  - The default number of iterations over which `pcg` batches the solution update (pstep) changed from 4 to 1 (x is updated every iteration, and only one vector of p history is kept).
    Memory for 3 vectors is saved, but `pcg` becomes a few % slower. Because the way the solution is accumulated changes, the results differ at the level of rounding
    (the number of iterations is the same). `PARCEL_PSTEP=4` at runtime or `-DPARCEL_PCG_NO_XHIST=0` at build time gives the previous behavior.
- The output of the solver execution time is now split into the main loop and its breakdown (dot / axpy / SpMV / preconditioner).

### Removed

- **`parcel::krylov::pcaarnoldi`** (communication-avoiding Arnoldi method) and the macros `CAARNOLDI_RESTART_S` / `_T` / `CAARNOLDI_QR_ALGO`.
  It had no restart and no convergence check, and the accuracy of Ritz values other than the few of largest absolute value was not guaranteed. The same use is
  replaced by `ks_arnoldi`.

### Fixed

- The solution update of `pcagmres` (`update_xvector`) used `MPI_COMM_WORLD` instead of the matrix communicator for the communicator of the work BLAS.
  It now passes the matrix communicator (`Mat.comm()->mpi_comm()`). The operations inside (gemv, axpy) do not communicate, so the results are unchanged.
- With the Fujitsu compiler (the host compiler of `omp.library-only` on A64FX; its clang mode is based on LLVM 7), the `#pragma clang fp contract(off)` for double-double arithmetic
  was not reverted at the end of `Include/ddlib.hpp`, and in the rest of the translation unit (all kernels: SpMV, axpy, inner products, preconditioners and so on)
  FMA instructions were not used (even with `-ffp-contract=fast`). This version does not know push / pop of `float_control`, and pop does not revert.
  The FMA prohibition is now limited to the bodies of the double-double functions (a pragma at the start of a compound statement, which in all versions of clang applies only within that compound statement).
  The generated code of the double-double functions is unchanged (the assembly matches between the Fujitsu compiler and clang 18). On A64FX, with the recommended flag (`-ffp-contract=fast`, see "Performance and Memory" below),
  the other kernels now use FMA, and the last digits of the results change. clang 11 and later (AdaptiveCpp for x86 and GPUs) already reverted on pop, so nothing changes there.
- The Fortran module (`capi/parcel_sycl_mod.f90`) can now also be compiled with Fujitsu Fortran (frt). `parcel_solve_linear` / `parcel_solve_eigen` are now
  Fortran procedures of the module (the names, the arguments and the optional `res` are the same; OPTIONAL arguments of BIND(C) procedures are a Fortran 2018 feature, which frt does not accept).
- `resize` / deallocation of `util_sycl::vector` also freed the memory of a non-owning view (`shallowcp`) pointing into another array.
- Device memory is now zero-initialized at allocation (when uninitialized values contained NaN / Inf, 0 × NaN broke the results;
  this showed up in single-precision, half-precision and double-double configurations).
- The `MPI_Wait` condition of the halo exchange (`SpMVcomm`) was swapped with the posting condition (causing incorrect waits for rank pairs with asymmetric send/receive counts).
- SpMV returned wrong results for partitionings in which the row blocks are not ordered by rank number.
- The point Jacobi preconditioner accessed out of range by the length of the halo.
- The shifted template arguments of `pbicgstab` (above). The type of the coefficient vectors is now `Tdot`, so it can be built with double-double.
- The double-double gemv dropped the low part of the coefficient vector (the solution update of pcg, and the orthogonalization of pgmres with CGS specified for QR).
- With double-double × the Jacobi preconditioner (CRS), the results were broken when matrix scaling was disabled.
- Out-of-range write in `tgemm` for A64FX.
- `ks_arnoldi`: in the restart with complex eigenvalues, the subspace contraction was not orthogonal (changed to the real Schur form). A problem in which a spurious
  invariant subspace with a low-precision basis could be reported as "converged" (now checked with the true residual, and if it cannot be confirmed the basis is expanded in a random direction).
  A problem in which it could not be built with mixed precision.
- The `ldvl` error of `LAPACKE_dgeev` in v1.1 (removed together with `pcaarnoldi`).
- When the initial guess `x0` is the exact solution (initial residual b − A x0 = 0), `pcg` returned `DIVERGED_NANORINF` and `pbicgstab` `DIVERGED_BREAKDOWN`.
  If the initial relative residual ‖b − A x0‖/‖b‖ is below `tol`, all linear solvers now return `CONVERGED_RTOL` with 0 iterations without iterating,
  and leave `x0` unchanged (`get_residual()` / `get_true_residual()` give the initial relative residual). When the initial residual was nonzero but below `tol`,
  `pcg` / `pbicgstab` iterated once before converging (number of iterations 1 → 0).
- The rank-deficiency threshold of `pcagmres` was overestimated in configurations that hold the basis in low precision (half- or single-precision basis). Even healthy columns
  were judged rank-deficient, and each cycle ran using only one column of the basis (the number of iterations increased greatly). The threshold is now the larger of the value determined by the computation precision of the QR
  (1e3 u, 1e2 √u for Cholesky QR) and the fluctuation of the basis storage precision 10 u_basis. When the basis and the QR computation have the same precision, the
  threshold is unchanged. Note that the Cholesky QR family (`CAGMRES_Q_IMPLICIT`) cannot be used with a half-precision basis (the "Precision" section of README.md).

- On GPUs, `pcagmres` with `CAGMRES_Q_IMPLICIT` + `QR_CAGMRES_CHOLESKYQR` (moment method) converged slowly or diverged for s = 4, 8.
  The matrix product used for the inner products (a dedicated path for 4 rows × an odd number of columns) included only some of the parallel partial sums in the result.

- The symmetric matrix product (used by Cholesky QR; the non-A64FX version) did not multiply C by beta and added it multiplied by the number of work-items
  (existing calls have C = 0 and beta = 0, so the results are unchanged).
- In the construction of the block Jacobi preconditioner for DIA matrices, the processing of each block copied the whole matrix (the result is the same; on GPUs the cost was the square of the number of rows).

- When one rank sees several GPUs and a non-default GPU is chosen for the matrix queue, the work areas for communication and inner products were allocated on a different GPU,
  and the run stopped with an illegal memory access. The work areas are now allocated with the queue of the matrix (or of the operation).
- In the construction of the block Jacobi preconditioner (CRS), a work area used only by a kernel for specific CPUs was allocated on other environments too (on GPUs, 256 times the number of rows in bytes).

### Performance and Memory

- Work arrays proportional to n that SpMV does not reference (1,792 bytes per point for the 7-point stencil) are no longer allocated.
  The problem size that fits on one GPU increased.
- Added the switch `_BLOCKING_LOOP_DOT_` for blocking of the reductions for CPUs (OpenMP backend)
  (avoids the fixed cost of barriers). The number of blocks is determined at runtime from the number of OpenMP threads (overridden by the environment variable `PARCEL_NBLOCK`).
- Added the switches `PARCEL_SPMV_LADDER` / `PARCEL_PRECON_LADDER`, which split the inner loops of SpMV and the preconditioner in a ladder shape for GPUs.
- `PARCEL_GEMV_AS_AXPY` for A64FX, which decomposes gemv into repeated axpy (default 1 in A64FX builds).
- Added `-ffp-contract=fast` to the recommended flags for A64FX (the "Build" section of README.md, `+fj` of the spack recipe, `wis_ody` / `wis_odyspack` in the example test settings `tests/site_config.py`).
  The clang mode of the Fujitsu compiler does not fuse multiply-adds into FMA instructions by default, and AdaptiveCpp appends `-O3` after `-Kfast`,
  so `-Kfast` had no effect either and no FMA appeared in the A64FX kernels. On one A64FX node (NX=256, pcg + Jacobi preconditioner) one iteration becomes 3 %
  faster (SpMV 10 %). The number of iterations is the same, and the last digits of the results change (about 1e-11 relative). FMA remains prohibited inside the double-double functions.
- Optimizations for A64FX (only the A64FX build `_BLOCKING_LOOP_`; the generated code of the x86 and GPU builds is unchanged; the "Build" section of README.md):
  - Added `OMP_WAIT_POLICY=ACTIVE` and `KMP_BLOCKTIME=infinite` to the recommended runtime settings (OpenMP threads do not sleep while waiting,
    reducing the fixed cost of kernel launches and waits; counterproductive with placements where cores are shared; set in `wis_ody` / `wis_odyspack` of the example test settings `tests/site_config.py`).
  - Rewrote axpy with coefficients passed by value (`axpy_value` / `axpby_value`) and the axpy-decomposed gemv (partitioning into contiguous ranges, copying pointers and coefficients into local variables,
    making explicit with `std::fma` the fusion the Fujitsu compiler chooses for the original expression, a beta = 1 path). `PARCEL_A64FX_AXPY_C` (default 1;
    `-DPARCEL_A64FX_AXPY_C=0` returns to the form before the rewrite; the results are bitwise identical either way).
  - `PARCEL_CG_FUSED` (default 1, disabled with `-DPARCEL_CG_FUSED=0`), which fuses the updates of x and p of `pcg` (pstep = 1), and the update of r with the inner product ‖r‖, each into one kernel.
  - The results are bitwise identical to before the rewrite. On A64FX (pcg + Jacobi preconditioner, one iteration of a fixed 200 iterations), NX=64 on 1 node is 0.59 times, NX=128 0.78 times,
    NX=256 0.99 times, and NX=512 on 8 nodes 0.84 times (relative to a build with A only).
  - `PARCEL_JACOBI_NODIV_UNIT` (default 1), which skips the division in the preparation of the right-hand side of the Jacobi preconditioner (D⁻¹b) when the divisor is 1 (no vector scaling;
    the default for double and single precision). The SVE division of A64FX is not pipelined and took about 30 % of the preconditioner.
  - `PARCEL_JACOBI_ROW2` (default 1; not used with a double-precision matrix, on-the-fly computation or `PARCEL_PRECON_LADDER=1`), which, when the preconditioner matrix is single precision (`precon_dataF` and so on),
    processes two rows alternately in the Jacobi preconditioner iterations to overlap the waits on multiply-add dependencies.
  - These two also give bitwise identical results (checked by the residual history up to convergence). On one A64FX node (NX=256, pcg + Jacobi preconditioner) the preconditioner section is
    0.77 times in double precision and 0.70 times with `precon_dataF`, and one iteration 0.90 times and 0.87 times.
- `-Xcuda-ptxas -maxrregcount=255` is recommended for V100 (sm_70) builds (the "Build" section of README.md). With ptxas's default register allocation,
  loads are split in kernels such as the Jacobi preconditioner, which becomes slower. There is no spill, and the results are unchanged. Added to the V100 environments in the example test settings (`tests/site_config.py`).
- Blocked the orthogonalization of `ks_arnoldi` and batched the allreduce operations.
- The inner products of the orthogonalization of `pcagmres` (R = Qᵀ V in the version that forms Q explicitly, and the Gram matrix of `CAGMRES_Q_IMPLICIT`) changed
  from a gemm (transposed) with one thread per output element to repeated local inner products (dot) + one allreduce
  (only as many threads as output elements were launched, which was slow on GPUs). The update V -= Q R in the version that forms Q explicitly is now a tall-skinny gemm with one thread per row
  (row blocks on A64FX) (each row of Q is read only once). The order of summation in the update is unchanged.
- **Reduction of work areas** (the expressions and order of operations are unchanged):
  - (D⁻¹)(L+U) of the Jacobi preconditioner (`PRECON_JACOBI`) shares the row pointers and column indices of the matrix A, and holds only the values in the same layout as A
    (0 at the diagonal positions) (the column indices and row pointers are no longer duplicated). The iterations skip the diagonal using the diagonal position recorded per row (1 byte per row).
    By default the values are not held either and are computed from the values of A at every preconditioning (see "Changed" above). The preconditioner must not outlive
    the matrix A. With 1 iteration (default), the 2 work vectors for the iterations are not allocated.
  - `pgmres`: work vectors whose live ranges do not overlap (residual and solution update, true residual) are placed in the same area, saving 4 vectors with halo.
  - `pcg`: the true residual is placed in the area of q (1 fewer). With the default pstep = 1 the p history is one vector, 3 fewer than with pstep = 4
    (see "Changed" above).
  - `pbicgstab`: work vectors whose live ranges do not overlap are placed in the same area, reducing the vectors with halo from 11 to 8.
  - `pcagmres`: in the version that forms Q explicitly, when the QR is MGS / CGS, the basis block V is built directly inside Q (the s+1 vectors of V are not allocated).
    The work vector of the solution update (with `CAGMRES_Q_IMPLICIT`, the vector at the start of the block) uses the residual area.
  - `ks_arnoldi`: the restart contraction V Z is done inside V, and the second basis (m+1 columns) is replaced by 3 columns for checking the true residual
    (when double-double is not involved and k_keep+1 <= 32).
- **Reduction of per-call fixed costs** (the expressions and order of operations are unchanged; with `-ffp-contract=off` the results are bitwise identical.
  In builds that allow multiply-add fusion, the way the compiler forms FMAs may change, changing rounding and the number of iterations):
  - The work areas of inner products, max and the Gram-matrix gemm are reused instead of being allocated at every call.
  - The coefficients of `pcg` / `pbicgstab` / `pgmres` (MGS) / `ks_arnoldi` (MGS) are passed to axpy by value instead of being written back to a one-element device array
    (combinations involving double-double are as before). Synchronization before and after the timing of inner products is omitted by default (`PARCEL_TIMER_SYNC`).
  - CPU (`_BLOCKING_LOOP_DOT_` / `_BLOCKING_LOOP_`): the "axpy of the previous column" and the "inner product of the next column" of MGS are fused into one kernel.
    The j+1 local inner products of CGS2 in `ks_arnoldi` are taken in one kernel. gemv and the tall-skinny gemm run per row block.
    64 bytes of padding are inserted between the columns of the Krylov basis (to avoid cache-set conflicts).
  - A64FX (`_BLOCKING_LOOP_`): SpMV and the Jacobi preconditioner are split into contiguous row ranges, and the arrays are copied into local variables with restrict.
    SpMV does not read y when beta = 0 (only the case where y is NaN / Inf and the sign of a −0 result can change).
