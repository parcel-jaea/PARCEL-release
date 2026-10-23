English | [日本語](README.ja.md)

# Regression Tests

Regression tests for PARCEL_SYCL v1.2. The verification sample `tests/verify/parcel_krylov_sample_verify.cpp` is built
for each precision configuration, solved with the solver, matrix, preconditioner and number of MPI processes switched by run-time options, and compared with a reference solution.
The job scripts are created by the generator `gen_regression.py` (to change the conditions, fix the generator and regenerate).

## Usage

Do everything at the root of the package (the generated scripts use the directory where `qsub` / `pjsub` was run
as the root of the package). With / without MPI and the language (verification sample (C++) / C / Fortran / Python) can be selected independently
at every stage of generation, build, submission and summary (see "Selecting What to Run" below). On machines without a scheduler, `run_local.sh` runs them (see below).

1. **Configure the environment.** Adapt `PROJECT` (account to be charged), `ENV_SETUP` (modules and environment variables)
   and `REGRESSION` (queues, resources, compile flags, launch commands) in `tests/site_config.py` to your environment.
   The 17 bundled environments (`s86_cpu` / `s86_gpu` / `wis_ody` / `wis_aqu` / `miy_c` / `miy_g` / `ps_a` / `ps_b`,
   `mac` without a scheduler, `s86spack_cpu` / `s86spack_gpu` which use s86 with a spack environment installed from scratch, and likewise
   `miy_cspack` / `miy_gspack` / `ps_aspack` / `ps_bspack` / `wis_aquspack` / `wis_odyspack` which test PARCEL installed with spack on other machines) are examples,
   so copy a close one and rewrite it.
2. **Generate.**
   ```sh
   python3 tests/regression/gen_regression.py [env ...]     # if omitted, all environments in REGRESSION and their configurations without MPI
   python3 tests/regression/gen_regression.py --plat s86_cpu --mpi both --lang cpp,c,fortran,python   # generate a selection
   ```
   The following are created in `tests/regression/` (when nothing is selected, all previous outputs are deleted before generating; when something is selected, only the outputs of the
   selected combinations are regenerated).
   - `sub_build_<plat>.sh`: builds the executables of the 13 precisions `bin/<plat>/verify_<precision>` and `bin/<plat>/ddcheck`
   - `sub_regression_<plat>_<precision>_<partition>_np<N>.sh`: test jobs (about 200 per environment)
   - `sub_regression_<plat>_ddcheck_np{1,4}.sh`: detection of double-double degradation
   - The same for the configuration without MPI `<plat>_nompi` (see "Configurations without MPI" below)
   - `sub_build_lang_<plat>.sh` and `sub_regression_<plat>_lang_{c,f,py}_np<N>.sh`: tests of the C / Fortran / Python APIs
     (see below; only for the environments in `REGRESSION_LANG` / `REGRESSION_LANG_PY`)
3. **Build.** Compilation uses neither MPI nor GPUs, so it may be run on a login node
   (environments that build on compute nodes, such as A64FX, submit it as a job).
   ```sh
   PARCEL_BUILD_PAR=6 bash tests/regression/sub_build_<plat>.sh   # number of concurrent compilations (default: the number of precisions)
   ```
   Existing executables are skipped, so if some fail, rerunning builds only the rest. To rebuild,
   delete `bin/<plat>/`. Additional flags can be given in `PARCEL_BUILD_EXTRA` (e.g. `PARCEL_BUILD_EXTRA=-ffp-contract=off`;
   the flags used are recorded in `bin/<plat>/build_extra.txt`, and the build stops if executables built with different flags would be mixed).
   When building several environments on the same login node, use `build_chain.sh` to build them one at a time.
   The compile flags are determined by `target` and `extra` in `site_config.py`. The V100 (sm_70) example `s86_gpu`
   adds `-Xcuda-ptxas -maxrregcount=255` (recommended on V100; it prevents ptxas register allocation from splitting the loads
   in the Jacobi preconditioners and elsewhere; no spills occur and the results do not change). The other GPU examples do not add it.
4. **Submit.**
   ```sh
   for f in tests/regression/sub_regression_<plat>_*.sh; do qsub $f; done
   ```
   If there is a limit on the number of jobs that can be submitted, use `feed_regression.sh` (submits one round) and `reg_drive.sh` (submits repeatedly).
   ```sh
   nohup bash tests/regression/reg_drive.sh <plat> <number of jobs to keep submitted> &
   ```
   Test jobs skip cases that have finished with OK, so resubmitting after a walltime overrun or the like runs only the rest
   (to start over from the beginning, use `PARCEL_REG_FORCE=1`).
5. **Summarize.**
   ```sh
   bash tests/regression/summarize.sh [<plat> ...]
   ```
   Prints the count of each status per environment and precision, the totals per language (C++ / C / Fortran / Python), and a list of cases other than OK.

## Selecting What to Run (with / without MPI, Languages)

With MPI (`<plat>`) / without MPI (`<plat>_nompi`) and the languages `cpp` (all precisions of the verification sample `verify_*` and ddcheck) / `c` / `fortran` / `python`
can be combined freely for generation, build, submission and summary. The builds are also separate per language and per with / without MPI, so, for example,
you can run only the C tests without building `verify_*`. The results are separated into `regression_results/<plat>[_nompi]/<precision | lang_c | lang_f | lang_py>/`.

| Stage | How to select |
|---|---|
| Generation | `gen_regression.py --plat <env,...> --mpi yes\|no\|both --lang cpp,c,fortran,python` (default: all environments, both, all languages). Only the outputs of the selected combinations are regenerated; other scripts are kept |
| Build | `cpp` uses `sub_build_<plat>.sh`; `c` / `fortran` / `python` use `sub_build_lang_<plat>.sh [c] [fortran] [python]` (the arguments select the languages; if omitted, all languages of that environment). Without MPI, use the `<plat>_nompi` scripts |
| Submission | `feed_regression.sh <plat> <limit> --mpi … --lang …` (runs the selected combinations in order; the limit is counted over all of them). One combination at a time with the kinds `cpp` / `lang_c` / `lang_f` / `lang_py` also works. `reg_drive.sh` passes its fifth and later arguments through as they are |
| Summary | `summarize.sh --mpi … --lang … [<plat> ...]` |

```sh
python3 tests/regression/gen_regression.py --plat s86_cpu --mpi no --lang c        # generate only C for s86_cpu_nompi
bash tests/regression/sub_build_lang_s86_cpu_nompi.sh c
bash tests/regression/feed_regression.sh s86_cpu 20 --mpi no --lang c
bash tests/regression/summarize.sh --mpi no --lang c s86_cpu
nohup bash tests/regression/reg_drive.sh s86_cpu 20 "" sel --mpi both --lang cpp,fortran &
```

The records of submitted jobs (`feed_<kind>_<plat>.jobs` / `.state`) are kept separately per kind, so do not submit the same job twice under a different kind
(the default `regression` includes both `cpp` and the per-language tests).

## Machines without a Scheduler (`sched="local"`, e.g. `mac`)

For environments with `sched="local"` in `REGRESSION` of `site_config.py`, the job scripts contain no resource specification, and `run_local.sh`
runs them directly with bash. The cases, judgments and result locations are the same as in the other environments. Jobs run concurrently are arranged so that the total number of
cores used (number of processes × number of threads) does not exceed `--cores` (default: the number of logical CPUs).

```sh
python3 tests/regression/gen_regression.py --plat mac          # mac and mac_nompi
bash tests/regression/run_local.sh --plat mac --mpi both --lang cpp,c,fortran,python --build   # build, run and summarize
```

- `--build` runs `sub_build_<plat>.sh` and `sub_build_lang_<plat>.sh` in order for each environment (two builds never run at the same time).
  The number of concurrent compilations is `PARCEL_BUILD_PAR` (default 4 in the example `mac`). `--no-run` only builds, and `--dry-run` only lists the jobs to run.
- Logs are in `tests/regression/local_logs/<plat>/<job name>.log`. Finished cases are skipped, so even if stopped, the same command continues from where it left off.
- When the time limit (`timeout`) is exceeded, only the children of that job are stopped (concurrent jobs using the same executable are not searched for by name and stopped).
- Example `mac` (macOS + Homebrew adaptivecpp, open-mpi, openblas and gcc; Python with numpy and scipy): 1 rank with 5 threads
  (the same as `s86_cpu`, so that the CPU block Jacobi partitioning is the same and the results can be compared; np=4 exceeds the number of cores, but
  `run_local.sh` runs jobs that exceed the number of cores on their own). Open MPI on macOS
  cannot bind processes to cores, so `--bind-to none` (`map_by=False`). There is no GNU `timeout`, so `ENV_SETUP["mac"]` provides a function
  with the same calling convention in perl. `run_local.sh` and the others work with macOS `/bin/bash` (3.2). The Homebrew location can be changed with `PARCEL_BREW`.

## Test Contents

| Item | Contents |
|---|---|
| Problem size | NX=8 (n=512). A dense-matrix reference solution can be built for all cases |
| Solvers | pcg / pbicgstab / pgmres / pcagmres / power / inverse_power / ks_arnoldi |
| Matrices | 3D 7-point Poisson, 3D convection-diffusion (nonsymmetric, c=0.05), convection-diffusion c=2.0 (eigenvalues are complex conjugate pairs; ks_arnoldi only) |
| Preconditioners | none / point Jacobi / block Jacobi / Jacobi |
| Number of processes | MPI 1 / 2 / 4 (a separate job per np; the resource specification matches np) |
| Precision configurations | 13 kinds (table below) |
| Restart length | The library defaults (pgmres 30, pcagmres (s, t) = (8, 4), ks_arnoldi maximum number of restarts 200). The exception is m = 15 for calcF16 × pgmres (below) |
| Others | The A M⁻¹ eigenvalue mode (shift-invert in `inverse_power` and `ks_arnoldi`), `pcagmres` without forming Q explicitly (`CAGMRES_Q_IMPLICIT`, CholeskyQR and the moment method; excluded for calcF16 / dataF16) |

The convection-diffusion matrix is tested only in double (the other precisions use Poisson only). `pcg` and `inverse_power` (which internally assume
a symmetric positive definite matrix) use Poisson only. `ks_arnoldi` × block Jacobi is heavy, so it is tested only in double.

The maximum number of iterations is 10000 for the linear solvers, 5000 for power and 200 for inverse_power. Only power × block Jacobi (eigenvalues of A M⁻¹) uses 20000,
because the eigenvalue gap changes greatly with the block boundaries (determined on CPUs by the number of threads per rank), and some partitionings reach 5000 iterations.

For the restart lengths, `PARCEL_GMRES_RESTART=default` / `PARCEL_CAGMRES_S=default` / `PARCEL_KS_RESTART=default`
are passed to the verification sample, and the tests use the library defaults without calling `set_restart` and the like (the compile-time values of the verification sample (pgmres 8, pcagmres (8, 1),
ks_arnoldi 50) are not used; the performance tests specify those values explicitly). Convergence is checked only at cycle boundaries, so the iteration count can
be larger than with a short cycle. Changes from the defaults for particular precisions are written in `STM_OVERRIDES` of `gen_regression.py`. Currently this is only
m = 15 for calcF16 × pgmres; with m = 30 the residual stagnates near the tolerance 1e-3 and the iteration count increases greatly.

### Precision Configurations and Thresholds

| Precision | Compile-time macros | Tolerance (outer, inner) | Error acceptance threshold |
|---|---|---|---|
| `double` | none | 1e-9 | 1e-6 (power 1e-4) |
| `calcDD` / `dataDD` | computation (and data) in double-double | 1e-9 | 1e-6 (power 1e-4) |
| `calcF` / `dataF` | computation (and data) in single precision | 1e-6 | 1e-4 (5e-4 for some preconditioned power cases) |
| `calcF16` | computation in half precision | 1e-3 | 2e-2 |
| `dataF16` | computation and data in half precision | 1e-2 | 5e-2 |
| `precon_calcF` / `precon_dataF` / `precon_allF` | preconditioner computation / data / both in single precision | 1e-9 (ks_arnoldi 1e-6 for calc / all) | 1e-6 (power 1e-4) |
| `precon_calcF16` / `precon_dataF16` / `precon_allF16` | preconditioner in half precision | 1e-9 (calc / all: pbicgstab 1e-5, ks_arnoldi 1e-3) | 1e-6 (1e-3 for pbicgstab, ks_arnoldi and power with calc / all, and for ks_arnoldi and power with data) |

The thresholds are matched to the relative residual reachable in each precision. In addition to the values in the table above, only for the combinations in which the residual hits its floor in low precision and breaks down or stagnates,
`LOWPREC_OVERRIDES` in `gen_regression.py` loosens the tolerance to about 10 times the measured floor (the error threshold is adjusted accordingly).

| Precision | Solver | Preconditioner | Environment | Tolerance | Error threshold |
|---|---|---|---|---|---|
| `precon_calcF16` / `precon_allF16` | pcg / pgmres / pcagmres (including implicit) | point Jacobi | all environments | 1e-5 | 1e-4 |
| `precon_calcF16` / `precon_allF16` | inverse_power | point Jacobi | all environments | inner 2e-5 | 1e-6 |
| `precon_calcF16` / `precon_allF16` | power | block Jacobi / Jacobi | all environments | 1e-5 | 2e-2 / 3e-3 |
| `precon_calcF16` / `precon_allF16` | A M⁻¹ mode (inverse_power, shift-invert) | point Jacobi / block Jacobi / Jacobi | all environments | inner 5e-2 | — |
| `precon_dataF16` | A M⁻¹ mode (inverse_power) | block Jacobi | all environments | inner 3e-2 | — |
| `calcF16` | inverse_power, A M⁻¹ mode | all kinds | all environments | inner 5e-2 | — |
| `calcF16` | pcagmres | all kinds | all environments | 1e-2 | — |
| `calcF16` | pbicgstab | point Jacobi / block Jacobi | all environments | 3e-2 | 1e-1 |
| `calcF16` | pcg | point Jacobi | all environments | 1e-2 | 2e-2 |
| `calcF16` / `dataF16` | pcg | block Jacobi | all environments | 1e-1 | 2e-1 |

"—" means the value in the table above is kept. The last 3 rows are combinations that break down or stagnate on GPUs; on CPUs (where half precision is emulated and the floor is lower)
they can also break down depending on the block Jacobi partitioning (number of threads per rank), so the same tolerance is applied on CPUs and GPUs.
The last row does not reach a practical accuracy and only confirms that it runs.

- `dataF16` also holds the vectors in half precision, so around a residual of 1e-2 the residual is of the same order as the rounding, and the recurrence residual departs from the true residual.
- A half-precision preconditioner can break down with NaN once the relative residual goes below ~1e-6 (especially in pbicgstab).
- The reference solution for preconditioned eigenvalues, A M⁻¹ itself, is affected by the rounding of the preconditioner, so for half-precision preconditioners the error threshold is matched
  to the relative precision of half precision (~1e-3).
- For `pcagmres` with half-precision data and computation (`calcF16` / `dataF16`), the version that forms Q explicitly (explicit) is tested,
  and the CholeskyQR family (implicit, moment method) is excluded (it gives NaN or does not converge regardless of s, t).
- The tolerance of the inner solvers (the inner solver of inverse_power, the inner pgmres of shift-invert) is the same as the outer one
  (except for the A M⁻¹ mode and the individual tolerances above).
  The Rayleigh quotient of a symmetric matrix has an accuracy of the square of the eigenvector error, so the inner tolerance need not be stricter than the outer one.
  Conversely, if the inner tolerance is too loose, a low-accuracy answer is returned as "converged".

## Verification Method

| Target | Reference solution |
|---|---|
| Linear solvers (Poisson) | Manufactured solution and direct solution by dense `dgesv` |
| Linear solvers (convection-diffusion) | Direct solution by dense `dgesv` |
| Eigenvalue solvers (no preconditioner) | Analytic eigenvalues |
| Eigenvalue solvers (point Jacobi) | Analytic eigenvalues times 1/d (when the diagonal is a constant d) |
| Eigenvalue solvers (block Jacobi, Jacobi, A M⁻¹ mode) | `dgeev` of the dense A M⁻¹ (built by applying the PARCEL preconditioner to unit vectors) |
| `inverse_power`, shift-invert | Analytic eigenvalues (λ(A) regardless of the preconditioner) |

## Results

Output goes to `regression_results/<plat>/<precision>/<case>_np<N>/`. `result.txt` is the standard output,
`parcel_result.txt` is the same content written by rank 0 (for MPI implementations that send the standard output to the job log),
`run_info.txt` is the node and time of the run, and `status.txt` is the judgment.

| `status.txt` | Meaning |
|---|---|
| `OK` | The run succeeded, converged, and the error from the reference solution is within the threshold |
| `NOT_CONVERGED` | Did not converge (`converged = NO` for eigenvalue solvers, final residual > tol for linear solvers, or the convergence state of the last solver run is `DIVERGED_*`) |
| `INNER_NOT_CONVERGED` | The inner linear solver of an eigenvalue solver reached the maximum number of iterations (`linsolve_failures` is 1 or more). Not counted as OK even if the outer iteration converged |
| `ERROR_TOO_LARGE` | The error from the reference solution exceeded the threshold |
| `NONFINITE` | NaN / Inf appeared in the solution, eigenvalues or residual |
| `VERIFY_SKIPPED` | The reference solution could not be built and the accuracy was not verified |
| `NO_VERIFY_OUTPUT` | There is no verification output (the output was not captured) |
| `RUN_FAIL_<rc>` | Run-time error (`RUN_FAIL_3` is a version mismatch: if `[VERSION] PARCEL <version>`, printed first by the verification sample, differs from `PARCEL_EXPECT_VERSION`, which the generator takes from `Include/parcel_version.hpp`, the sample exits without testing) |
| `TIMEOUT` | Aborted (300 seconds for ks_arnoldi and power, 120 seconds for the others) |
| `BUILD_MISSING` | The executable does not exist (build first) |

Convergence of the linear solvers is judged by the recurrence residual used in the stopping criterion (the last `N : residual norm` in the iteration history).
The true residual `b-Ax` departs from the recurrence residual in low-precision configurations, and in pcg, when the interval at which solution updates are batched (`PARCEL_PSTEP`, default 1)
is 2 or more, it is meaningful only at that interval, so it is not used for the judgment. The accuracy of the solution itself is checked by the error from the reference solution.

### Detecting double-double Degradation (`ddcheck`)

The acceptance thresholds of the double-double configurations (`calcDD` / `dataDD`) are at the double level, so even if
double-double silently falls back to double because of compiler optimization or the like, the tests pass. `ddcheck/ddcheck.cpp` feeds inputs whose exact values are known and
cannot be represented in double to the same `parcel::blas` paths as the dd configurations of the solvers (dot, axpy, gemv, SpMV, residual computation; including MPI reduction and halo exchange),
and checks whether the relative error is 1e-28 or less. Only `[DDCHECK] RESULT PASS` gives `OK`;
FAIL gives `DD_DEGRADED`. `sub_build_<plat>.sh` builds `bin/<plat>/ddcheck` with the same flags as calcDD.
`gemv_coef_lo` (gemv with a dd coefficient vector) is not counted toward pass/fail by default (it is counted with `DDCHECK_STRICT_GEMV=1`).

## Configurations without MPI (`<plat>_nompi`)

A configuration that runs an executable built with `-DPARCEL_MPI=0` (which neither includes nor links MPI) directly as one process, without
`mpirun` / `mpiexec`, is also tested. For each environment listed in `REGRESSION_NOMPI` of `tests/site_config.py`
(default: all environments in `REGRESSION`), scripts with the environment name `<plat>_nompi` are created. The settings are derived from the original environment,
and only the following points differ.

| Item | Configuration without MPI |
|---|---|
| Compiler | The `CXX=` line in `ENV_SETUP` is replaced by `NOMPI_CXX` (default `acpp`). The MPI wrapper is not used |
| Compile flags | `-DPARCEL_MPI=0` is added to the flags of the original environment |
| Executables | `bin/<plat>_nompi/verify_<precision>` and `bin/<plat>_nompi/ddcheck` (built by `sub_build_<plat>_nompi.sh`) |
| Number of processes | 1 only (test jobs are `_np1` only, and ddcheck is np=1 only). The resource specification is the same as np=1 of the original environment |
| Launch | The executable is launched directly. On GPUs, the binding script is not used, and the same device as rank 0 of np=1 with MPI (`CUDA_VISIBLE_DEVICES=0` / `HIP_VISIBLE_DEVICES=0`) is used. NUMA pinning of ranks (`pin`) is not used |
| Cases, judgment | The same as np=1 of the original environment |
| Results | `regression_results/<plat>_nompi/` |

Build, submission and summary follow the same steps as for the environments with MPI; just specify `<plat>_nompi` as the environment name
(`feed_regression.sh <plat>` does not include the `<plat>_nompi` scripts).

```sh
bash tests/regression/sub_build_s86_cpu_nompi.sh
nohup bash tests/regression/reg_drive.sh s86_cpu_nompi <number of jobs to keep submitted> &
bash tests/regression/summarize.sh s86_cpu_nompi
```

Without MPI and with MPI np=1 perform the same computation, so the results should agree. Compare them with `compare_nompi.py`.

```sh
python3 tests/regression/compare_nompi.py s86_cpu [...] [--md compare_nompi.md]   # on a host that has both results
python3 tests/regression/compare_nompi.py --csv iters_*.csv                     # from the CSV of collect_iters.sh
```

It pairs `regression_results/<plat>/.../<case>_np1` with `regression_results/<plat>_nompi/.../<case>_np1` (including ddcheck),
counts whether the state, iteration counts (the last number in the iteration history, the outer iteration count and the number of restarts), numerical values (error from the reference solution, true residual, eigenvalues, `[DDCHECK]` lines)
and the whole output (excluding the time lines and AdaptiveCpp warnings) agree, and lists the pairs that differ. If the state, iteration counts or numerical values differ for even one pair,
the exit code is 1. When comparing from CSV, only the state and iteration counts are checked.

## Tests of the C / Fortran / Python APIs (`lang_c` / `lang_f` / `lang_py`)

`examples/sample_c.c` / `examples/sample_f.f90` (examples that call all solvers through the C / Fortran APIs and verify the results independently) and
`tests/regression/sample_py.py` (which does the same problems, items and judgments from the Python package `parcel`) are put into the regression tests as they are.
The targets are the environments listed in `REGRESSION_LANG` (C / Fortran) and `REGRESSION_LANG_PY` (Python) of `tests/site_config.py`, and their configurations without MPI.

| Item | Contents |
|---|---|
| Build | `sub_build_lang_<plat>.sh [c] [fortran] [python]` builds `bin/<plat>/sample_c`, `sample_f` and `libparcel_py.so` (may be run with bash on a login node; if the arguments are omitted, all languages of that environment). The flags are the same as for `verify_*` (`target`, `extra`, `PARCEL_BUILD_EXTRA`, and `-DPARCEL_MPI=0` without MPI). For C / Fortran, the wrapper `capi/wrapper_parcel_sycl.cpp` is compiled with the SYCL compiler and main with `LANG_MPI_CC` / `LANG_MPI_FC` (`LANG_NOMPI_CC` / `LANG_NOMPI_FC` without MPI), and they are linked with the SYCL compiler (the same form as `examples/build_examples.sh`). Fortran without MPI is combined with `examples/sample_f_mpi_stub.f90`. For Python, the native layer `python/src/parcel_py.cpp` is built as a shared library (pip is not used), and the Python with numpy and scipy (`PARCEL_REG_PY`, default `python3`) is recorded in `bin/<plat>/lang_python.txt` |
| Jobs | `sub_regression_<plat>_lang_{c,f,py}_np{1,2,4}.sh` (only `_np1` without MPI). Resources, launch and number of threads are the same as in the other tests. NX is the sample default of 16. Python reads the package's `python/` and `bin/<plat>/libparcel_py.so` (`PARCEL_TEST_FROM_SOURCE=1`, `PARCEL_PY_LIB`) |
| Judgment | 1 item = 1 case. `regression_results/<plat>/lang_<c\|f\|py>/<item>_np<N>/status.txt` is `OK` for PASS and `FAIL` for FAIL. The item name has symbols replaced with `_` (`pcagmres(impl,sym)` → `pcagmres_impl_sym`). The final `ALL PASS` line together with exit code 0 counts as 1 case (`all`), so one run gives 13 cases for C / Fortran (the leading `version` is the version check: whether `parcel_get_version()` and `parcel_version_string()` give the same version and match `PARCEL_EXPECT_VERSION`, which the generator takes from `Include/parcel_version.hpp`). Python checks `parcel.solve` with the settings omitted (`pcg_defaults`) instead of the items of the old API, and adds at the beginning a check of whether the native layer has MPI (`has_mpi`) and, in GPU environments (environments with `py_device` written), the device name (`device`). `version` is the agreement of `parcel.library_version()` (native layer), `parcel.__version__` and `PARCEL_EXPECT_VERSION` (14 cases, 15 on GPU). Items missing from the output (crashed midway) are `TIMEOUT` / `RUN_FAIL_<rc>` / `NO_ITEM_OUTPUT` |
| Output | The whole output is in `lang_<language>/_run_np<N>/result.txt` (no status.txt is placed there, so it is not counted). The `result.txt` of each case contains only the relevant lines |

```sh
PARCEL_REG_PY=<python with numpy and scipy> bash tests/regression/sub_build_lang_s86_cpu.sh
bash tests/regression/feed_regression.sh s86_cpu 20 lang     # waits for bin/s86_cpu/lang_built.txt and submits (for one language only: lang_c / lang_f / lang_py)
bash tests/regression/summarize.sh s86_cpu                   # lang_c / lang_f / lang_py rows and the totals per language
```

`feed_regression.sh <plat>` (the default regression) also includes the `lang` jobs (as with ddcheck). Run `sub_build_lang_<plat>.sh`
beforehand (otherwise each case becomes `BUILD_MISSING`). The 2 Wisteria environments, whose C / Fortran / Python tests were not checked during development,
are not included in `REGRESSION_LANG` / `REGRESSION_LANG_PY` (`wis_aquspack` / `wis_odyspack`, which use PARCEL installed with spack, are included).

## Running in a spack Environment (`s86spack_cpu` / `s86spack_gpu`, `miy_cspack`, etc.)

An example of running the tests with the compilers and libraries of a spack environment installed from scratch with `spack/stack/install_stack.sh` (`spack/README.md`).
The queues, resources, flags and cases are the same as `s86_cpu` / `s86_gpu`; only the environment setup of the jobs (`ENV_SETUP["s86spack"]`) differs.

```sh
python3 tests/regression/gen_regression.py --plat s86spack_cpu,s86spack_gpu --mpi both
export PARCEL_STACK=$HOME/parcel_stack                     # install location of install_stack.sh (default: SPACK_STACK in site_config.py)
bash tests/regression/sub_build_s86spack_cpu.sh            # the script loads the spack environment itself
bash tests/regression/sub_build_lang_s86spack_cpu.sh       # Python is the python3 of the spack environment (with numpy and scipy)
for f in tests/regression/sub_regression_s86spack_cpu_*.sh; do qsub -v PARCEL_STACK $f; done
```

- The build scripts and jobs activate the spack environment with `. ${PARCEL_STACK}/setup.sh` and load AdaptiveCpp, LLVM,
  OpenBLAS and Open MPI with `spack load hipsycl openblas openmpi`. LAPACK is linked from spack's OpenBLAS (`${openblas_root}`).
- What to test is selected with `parcel_from`.
  - `"source"` (the default of `parcel_from`): builds the package sources with the spack toolchain. The verification sample (C++) and ddcheck use
    `Include/`, C / Fortran use the wrapper in `capi/`, and Python uses `python/`.
  - `"installed"`: tests PARCEL installed with spack. C / Fortran link with `libparcel_sycl_c.so` / `libparcel_sycl_f.a` of `parcel-sycl` and
    its headers and modules, and Python loads `py-parcel-sycl` with `import parcel` (`spack load` with the configuration written in `spack_parcel` and with / without
    MPI). The verification sample (C++) and ddcheck are built with the headers of `parcel-sycl` (`$PARCEL_PREFIX/include`)
    (the sample reads `Include/` by a relative path, so the build script copies the test sources to `buildlog/<plat>/installed_src/` and
    points the `Include` next to them at `$PARCEL_PREFIX/include`; the location used is in `bin/<plat>/parcel_prefix.txt`).
- The bundled examples (`s86spack_*` and, on other machines, `miy_cspack` / `miy_gspack` / `ps_aspack` / `ps_bspack` / `wis_aquspack` / `wis_odyspack`) use `"installed"` and
  test PARCEL installed with spack itself. The queues, resources, flags and cases are the same as the original environments (`s86_cpu`, `miy_c`, etc.). The environment setup outside s86 is `_stack_env` in `site_config.py`
  (`setup.sh` loads the site modules, then `spack load hipsycl openblas`), and the install location is `SPACK_STACK_SITE` (the job's `PARCEL_STACK` takes precedence).
  ```sh
  python3 tests/regression/gen_regression.py --plat miy_cspack --mpi both
  bash tests/regression/sub_build_miy_cspack.sh; bash tests/regression/sub_build_lang_miy_cspack.sh   # likewise without MPI (_nompi)
  nohup bash tests/regression/reg_drive.sh miy_cspack 4 "" sel --mpi both --lang cpp,c,fortran,python &
  ```
- Wisteria-Odyssey (`wis_odyspack`, A64FX) differs from the others in the following points. The environment setup is `ENV_SETUP["odyspack"]` (not `_stack_env`).
  - Builds are done on compute nodes (the login nodes are x86). `pjsub tests/regression/sub_build_wis_odyspack.sh` (and `_nompi`, `sub_build_lang_*`).
  - PARCEL is `parcel-sycl+fj`, built with the Fujitsu compilers, MPI and LAPACK. The verification sample is built with `mpiFCC`, as in `wis_ody` (the
    host side of AdaptiveCpp is FCC), and the C / Fortran tests are built with `mpifcc` / `mpifrt` (`fcc` / `frt` without MPI) with `-Kopenmp`
    (the installed Fortran modules were built with frt and cannot be read by gfortran; when PARCEL is called with multiple threads from a program built with frt
    without `-Kopenmp`, the results of the linear solvers are corrupted). The output directory of frt modules is `-M` (`lang_fmod`).
  - Fujitsu `mpiexec` writes the standard output of ranks to the job log, not to the shell redirection target (`> result.txt`).
    `-std mpiexec_out.txt` is added to the launch command and appended to `result.txt` after the run (`rank_out`). This is needed because the C / Fortran / Python tests judge from the standard output.
  - Fujitsu MPI cannot initialize MPI without going through `mpiexec`, so the check of `import parcel` with MPI in the per-language build also
    goes through `mpiexec -n 1` (`py_check_launch`).
- The performance tests can also be run in the environments of the same names (`PERF["s86spack_cpu"]` etc. in `tests/site_config.py`) (there is no `PERF` for the spack environments on other machines).

## Comparing Iteration Counts across Environments (Report Only)

Pass/fail is judged only by error and convergence; differences in iteration counts do not cause failures. In low-precision configurations, rounding differences due to the order of reductions (which changes with the number of threads, the number of ranks and the
device) are amplified and the iteration counts vary between environments, so their magnitude is reported separately.

Even in the same environment, the order of reductions changes with the number of nodes the ranks are spread over, and so do the iteration counts of low-precision configurations.
The test jobs therefore place the ranks on the smallest number of nodes. On PS-B (`ps_b` / `ps_bspack`; 1 rank per chunk, 1 node = 4 APUs, `node_chunks=4`),
np up to 4 uses `place=pack` (one node), and np above 4 uses `place=exclhost` (exclusive hosts) and `ppr:4:node` in the launch command, which puts
4 ranks on each of ceil(np/4) nodes (`place` and `node_chunks` in `tests/site_config.py`). Without `place`, the queue default (free on B_S) scatters the chunks
over free vnodes, and even np=4 can be spread over 2 to 3 nodes.

```sh
bash tests/regression/collect_iters.sh <plat> [...] > iters_<host>.csv          # on each host
python3 tests/regression/compare_iters.py iters_*.csv --md compare_iters.md     # locally
```

## Files

| File | Contents |
|---|---|
| `gen_regression.py` | Generator of the job scripts (definitions of precision configurations, cases, thresholds and judgments) |
| `ddcheck/ddcheck.cpp` | Detection of double-double degradation |
| `sample_py.py` | Tests of the Python API (following `examples/sample_c.c`; `lang_py`) |
| `feed_regression.sh` | Submits one round while keeping the number of jobs submitted at the same time (`regression` / `cpp` / `lang` / `lang_c` / `lang_f` / `lang_py` / `dd` / `pspec` / `perf` / `prec`, or `--mpi` / `--lang`) |
| `reg_drive.sh` | Calls `feed_regression.sh` repeatedly |
| `build_chain.sh` | Builds multiple environments one at a time |
| `summarize.sh` | Summary of the results (per environment and precision, per language; can be narrowed with `--mpi` / `--lang`) |
| `run_local.sh` | Builds, runs and summarizes on machines without a scheduler (`sched="local"`) |
| `collect_iters.sh`, `compare_iters.py` | Comparison of iteration counts across environments |
| `compare_nompi.py` | Comparison of the configuration without MPI with np=1 with MPI |
