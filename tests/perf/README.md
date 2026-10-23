English | [日本語](README.ja.md)

# Performance Tests

Performance tests for PARCEL_SYCL v1.2. Whereas the regression tests (`../regression/`) check "correctness", these check "speed".
The time to solve each solver to convergence is measured with 4 / 8 / 16 devices.
The executables are `bin/<plat>/verify_<precision>`, built by the regression test build (`tests/regression/sub_build_<plat>.sh`)
(the compile-time NX is used only as the default of the run-time option `PARCEL_NX`, so the same executables can be used for the measurements).

## Usage

Run at the root of the package. The settings for each environment are in `PERF` of `tests/site_config.py` (queues, resources, parallel layout, walltime).

```sh
python3 tests/regression/gen_regression.py --plat <plat> --lang cpp   # create the build script (shared with the regression tests)
bash tests/regression/sub_build_<plat>.sh                             # build bin/<plat>/verify_*

python3 tests/perf/gen_perf.py [--nx 512] [<plat> ...]     # tests/perf/sub_perf_<solver>_d<devices>_<plat>.sh
python3 tests/perf/gen_perf_prec.py [<plat> ...]           # tests/perf/sub_perf_d<devices>_<plat>_prec_<precision>.sh and sub_build_prec_<plat>.sh
bash tests/perf/sub_build_prec_<plat>.sh                     # build the mixed-precision executables bin/<plat>/perf_prec/verify_<precision>
for f in tests/perf/sub_perf_*_<plat>.sh tests/perf/sub_perf_*_<plat>_prec_*.sh; do qsub $f; done
python3 tests/perf/parse_perf.py . [--csv perf.csv]        # summary (fastest of 3 runs each)
python3 tests/perf/parse_perf_prec.py . [--md prec.md]     # comparison of mixed precision with double precision
```

If there is a limit on the number of jobs that can be submitted, use `bash tests/regression/feed_regression.sh <plat> <limit> perf` (`prec` for mixed precision) or
the kinds `perf` / `prec` of `reg_drive.sh`. If `--nx` is other than 512, `_nx<NX>` is appended to the script names and the output directory
(submit with `feed_regression.sh <plat> <limit> perf _nx<NX>`, summarize with `parse_perf.py . --dir perf_results_nx<NX>`).

## Measurement Conditions

| Item | Setting |
|---|---|
| Problem | 3D 7-point grid NX=512 (n = 134,217,728), b = 1, x0 = 0, double precision |
| Matrix | Convection-diffusion (`PARCEL_MATRIX=convdiff PARCEL_CONVDIFF_C=0.05`) for the solvers for nonsymmetric matrices (pgmres, pcagmres); Poisson for pcg, pbicgstab and the eigenvalue solvers |
| Convergence | Down to a relative residual of 1e-8, maximum 10,000 iterations |
| Settings | Solver parameters are the library defaults (`PARCEL_LIB_DEFAULTS=1`). Only the values in the table below are specified explicitly |
| Runs | 3 runs per configuration. The summary uses the run with the shortest total time |
| Number of devices | 4 / 8 / 16 (a socket on CPUs, one GPU on GPUs) |

| Job | Configuration (label) | Preconditioner | Matrix |
|---|---|---|---|
| `pcg` | `pcg_p{0,4}_*` | none / Jacobi | Poisson |
| `pbicgstab` | `pbicgstab_p{0,4}_*` | none / Jacobi | Poisson |
| `pgmres` | `cd0.05_pgmres_m60_p{0,4}_*` (restart m=60) | none / Jacobi | convection-diffusion |
| `pcagmres_explicit` | `cd0.05_pcagmres_explicit_s8t8_p{0,4}_*` (s = t = 8, forms Q explicitly) | none / Jacobi | convection-diffusion |
| `pcagmres_implicit` | `cd0.05_pcagmres_implicit_s8t8_p{0,4}_*` (s = t = 8, CholeskyQR version that does not form Q) | none / Jacobi | convection-diffusion |
| `power` | `power_p0_NONE` (maximum 20,000 iterations) | none | Poisson |
| `inverse_power` | `inverse_power_p0_NONE` (outer maximum 200, inner solver pcg with Jacobi preconditioner (`PARCEL_LINSOLVE_SOLVER=pcg PARCEL_LINSOLVE_PRECON=4`), inner tolerance 1e-10, maximum 5,000) | none | Poisson |
| `ks_arnoldi` | `ks_arnoldi_p0_NONE` (m = 40, k_keep = 10, nev = 1, maximum 200 restarts) | none | Poisson |

- pbicgstab breaks down on convection-diffusion without a preconditioner at NX=512 (a property of the method due to the amplification of rounding errors; the same happens with PETSc's BiCGStab), so
  it is measured on Poisson.
- The eigenvalue solvers are limited to no preconditioner because with a preconditioner they would solve a different problem, the eigenvalues of A M⁻¹.
- The orthogonalization of `ks_arnoldi` is measured with the library default (CGS2). However, PS-A (`ps_a`) is measured with MGS2 (`PARCEL_KS_QR=mgs`; the verification sample
  calls `set_qr_algo(QR_MGS)`) (the `ps_a` build also defaults to MGS2 (`-DPARCEL_KS_ORTHO_DEFAULT=mgs`), but it is also specified explicitly with the run-time option),
  because MGS2 can be faster on CPUs with a small local length per thread and a large cache. Run-time options added per environment are
  written in `job_args` (`{job name: run-time options}`) of `PERF` in `site_config.py` (check
  `qr = MGS (env)` in the `[LIB_DEFAULTS]` line of the first run's output).
- In the output of the first run of each configuration, the `[LIB_DEFAULTS]` line is used to check that the run-time options took effect (if not, the job stops with `PARAM_NG`).
  Performance measurement (`PARCEL_VERIFY=0`) does not print the matrix name, so at NX=512 the first 4 digits of the residual history are used to loosely check
  whether it is convection-diffusion or Poisson (a mismatch does not stop the measurement, but leaves `soft-NG` in `param_check.txt`).

## Mixed-Precision Comparison (`gen_perf_prec.py`)

Under the same conditions and parallel layout, only the executable is replaced with `bin/<plat>/perf_prec/verify_<precision>`, and pcg with the Jacobi preconditioner (Poisson) and
pgmres m=60 (convection-diffusion) are measured. The default precision is `precon_dataF` (the data of the preconditioner matrix and preconditioner vectors in single precision).
Names of the regression test precision configurations can be listed, as in `--prec precon_dataF,precon_dataF16`.

The executables are built with `tests/perf/sub_build_prec_<plat>.sh`, written by `gen_perf_prec.py` (the same environment, compile line and precision flags as the regression test build
`sub_build_<plat>.sh`, plus `PERF_EXTRA_FLAGS` of `gen_perf_prec.py`).
**`precon_dataF` is built without on-the-fly computation of the Jacobi preconditioner (`-DPARCEL_JACOBI_ONTHEFLY=0`).** With the default (on-the-fly computation enabled,
`PARCEL_JACOBI_ONTHEFLY=1`; the "Memory and Speed" section of README.md), even if the preconditioner matrix is in single precision, each preconditioner application reads the double-precision
values of the original matrix A, so the benefit of single-precision data hardly appears. Without on-the-fly computation, the preconditioner matrix (D⁻¹)(L+U) is stored and
read in single precision. Memory increases accordingly (4 bytes per nonzero of A; with a 7-point grid of NX=512 and 4 devices, about 0.9 GB per device).
The double precision being compared against (the results of `gen_perf.py`, `bin/<plat>/verify_double`) and `precon_dataF` of the regression tests
(`bin/<plat>/verify_precon_dataF`) remain at the default (on-the-fly computation enabled). The macros after preprocessing of the build are
kept in `bin/<plat>/perf_prec/macro_<precision>.txt` (`#define PARCEL_JACOBI_ONTHEFLY 0`).
The label is `<double-precision label>_<precision>`, and the output directory is the same as for the performance tests. `parse_perf_prec.py` compares with the
double precision (results of `gen_perf.py`) of the same environment and the same number of devices, and reports the ratio of total time (including the difference in iteration counts) and the ratio per iteration. The target environments are `PERF_PREC_PLATS`.

## Parallelism

| Example environment | 1 device | np (4 / 8 / 16 devices) | Number of nodes |
|---|---|---|---|
| `s86_cpu` | 1 socket = 4 ranks x 5 threads | 16 / 32 / 64 | 2 / 4 / 8 |
| `s86_gpu` | 1 V100 = 1 rank x 12 threads | 4 / 8 / 16 | 1 / 2 / 4 |
| `miy_c` | 1 socket = 4 ranks x 14 threads (1 rank per NUMA) | 16 / 32 / 64 | 2 / 4 / 8 |
| `miy_g` | 1 GH200 = 1 rank x 72 threads | 4 / 8 / 16 | 4 / 8 / 16 |
| `ps_a` | 1 socket = 9 ranks x 14 threads (3 ranks per NUMA) | 36 / 72 / 144 | 2 / 4 / 8 |
| `ps_b` | 1 MI300A = 1 rank x 6 threads | 4 / 8 / 16 | 1 / 2 / 4 |

- On CPUs, ranks are placed per socket with `mpirun --map-by ppr:<rpd>:socket:PE=<threads> --bind-to core`.
- On GPUs, the number of ranks per node is "the smaller of np and the number of GPUs per node"
  (beyond that, ranks that cannot get a GPU run on the CPU, and collective communication limits the whole). Each rank is assigned one GPU (`gpu_bind_*.sh`).
- `gpu_bind_*.sh` selects the GPU by the order within the node (local rank), so if the job's devices share a node with other jobs,
  it uses GPUs that were not assigned. Environments where 1 device = 1 chunk (e.g. `ps_b`) use `place` to make jobs that fit in one node
  `pack` and jobs that do not fit `exclhost` (exclusive use of the nodes).
- Intel MPI does not interpret `--map-by`, so `-ppn` sets the number of ranks per node, and masks per NUMA are listed in `I_MPI_PIN_DOMAIN`
  for pinning (`PIN_SETUP` in `gen_regression.py`; `ps_a` in the examples).
- `OMP_PROC_BIND` is not set (libomp binds the initial thread to one core, and the worker threads that inherit it
  lose their parallelism).
- The walltime is determined from the per-environment default and per-solver, per-device-count overrides (`walltime_by`), and the first queue in `queues` that fits is used.
  The bundled values are estimates from measurements during development, so adjust them to your environment. The 2 Wisteria environments were not measured under these conditions, so they are not in `PERF`.

## Pass Run-Time Options as Command-Line Arguments

```sh
mpirun -n 4 ./bin/<plat>/verify_double PARCEL_NX=512 PARCEL_SOLVER=pcg PARCEL_PRECON=4 PARCEL_LIB_DEFAULTS=1
```

The verification sample also accepts run-time options as environment variables, but depending on the MPI implementation, environment variables may not reach ranks on other nodes,
and only those ranks run with the default values and break down. Command-line arguments always reach all ranks.
Environment variables take precedence over arguments, so if `PARCEL_*` remains in the job's environment, a notice is printed at the start of the job.

## Results

Output goes to `perf_results/<plat>/<configuration>_np<N>/`.

| File | Contents |
|---|---|
| `status.txt` | `OK` / `TIMEOUT` / `RUN_FAIL_<rc>` / `PARAM_NG` / `PIN_IGNORED` |
| `result<R>.txt` | Output written by rank 0 for each run (`PARCEL_OUTPUT`). The summary reads this preferentially |
| `run<R>.txt`, `time<R>.txt` | Standard output and elapsed time [s] of each run |
| `dev.txt` | Number of devices |
| `param_check.txt` | Check of the first run's output (the `[LIB_DEFAULTS]` line and the matrix) |
| `bindings.txt` | Which socket and node each rank was placed on |

For the run with the shortest total time out of 3, `parse_perf.py` picks up the total time, the iteration count, the time per iteration [ms], and the breakdown into `init` / `loop` / `dot` /
`spmv` / `axpy` / `precon` / `qr` (`--csv`). Configurations that did not converge even though `status.txt` is OK are shown as "not converged".
Configurations whose longest of the 3 runs exceeds 1.5 times the shortest are marked with `※`.
`dot` includes `MPI_Allreduce` and `spmv` includes the halo exchange, so they reflect communication performance, while `axpy` is computation only,
so communication problems can be separated from computation problems. By default the device is not synchronized before and after timing the inner products, so `dot` includes the time of the preceding
kernel (such as the axpy of MGS). To obtain an accurate breakdown, add `PARCEL_TIMER_SYNC=1` at run time
(the time of the whole solver increases slightly).

## Notes

- **Concurrently running jobs do not write to the same files.** The GPU binding scripts (`gpu_bind_*.sh`) have
  a separate name for each job.
- **Staleness of the executables also checks the headers.** PARCEL is header-only, so rebuild after modifying `Include/`
  (jobs print a warning if `Include/*.hpp` is newer).
- **Finished configurations are skipped.** Resubmitting after a walltime overrun or the like measures only the rest. To measure again, use `PARCEL_PERF_FORCE=1`.

## Files

| File | Contents |
|---|---|
| `gen_perf.py` | Generator of the performance test job scripts (definitions of measurement conditions and configurations) |
| `gen_perf_prec.py` | Generator of the job scripts and build script (`sub_build_prec_<plat>.sh`) for the comparison of mixed precision with double precision |
| `parse_perf.py` | Summary of the performance tests |
| `parse_perf_prec.py` | Summary of the comparison of mixed precision with double precision |
