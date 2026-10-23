English | [日本語](README.ja.md)

# Installing with spack

`spack/` contains what is needed to install PARCEL_SYCL with spack.

| Location | Contents |
|---|---|
| `spack_repo/parcel/` | Package repository for spack (1.x, repository API v2.2). Recipes for `parcel-sycl` and `py-parcel-sycl` |
| `stack/install_stack.sh` | Script that installs everything from scratch, from fetching spack through LLVM, AdaptiveCpp, OpenBLAS and Python to PARCEL |
| `stack/check_stack.sh` | Checks what was installed (as far as can be done on a login node) |
| `stack/envs/<env>/` | Per-environment spack definition (`spack.yaml`) and additional settings |

## Recipes

| Package | Contents |
|---|---|
| `parcel-sycl` | Headers (`include/`), the C API `lib/libparcel_sycl_c.so` and `include/parcel_sycl_c.h`, the Fortran module `include/parcel_sycl_mod.mod` and `lib/libparcel_sycl_f.a` |
| `py-parcel-sycl` | The Python package `parcel` (builds `python/` with scikit-build-core) |

Variants (common to both):

| Variant | Default | Meaning |
|---|---|---|
| `mpi` | off | With MPI (the default is the single-process version without MPI) |
| `cuda` + `cuda_arch=<n>` | off | AdaptiveCpp `--acpp-targets=cuda:sm_<n>`. With `cuda_arch=70` (V100), `-Xcuda-ptxas -maxrregcount=255` is added automatically |
| `rocm` + `amdgpu_target=<gfx>` | off | `--acpp-targets=hip:<gfx>` |
| (neither) | | `--acpp-targets=omp` (CPU) |
| `native` | on | `-march=native -mtune=native` |
| `kernel` | auto | Kernel optimization: auto (blocking_dot on CPUs, none on NVIDIA GPUs (`+cuda`), ladder on AMD GPUs (`+rocm`), a64fx on A64FX), blocking_dot, a64fx, ladder, none |
| `fj` | off | A64FX: compiles the host side with the Fujitsu compiler (FCC) and builds the Fortran module with frt (confirmed with the Wisteria-Odyssey definition `envs/wis_ody`). Adds `-std=c++17 -ffp-contract=fast` (the clang mode of FCC does not fuse multiply-adds into FMA by default; the "Build" section of README.md) |
| `ks_ortho` | cgs2 | Default orthogonalization of `ks_arnoldi` (when the program does not call `set_qr_algo()`). `mgs` adds `-DPARCEL_KS_ORTHO_DEFAULT=mgs` (also included in `PARCEL_CXXFLAGS`). `envs/psa` (PS-A) uses `ks_ortho=mgs` |
| `capi` (`parcel-sycl` only) | on | Builds the C / Fortran APIs |
| `scipy` (`py-parcel-sycl` only) | on | Also installs SciPy (used by `DistributedCSR.from_scipy` / `from_global` and `examples/quick_start.py`; corresponds to `parcel[scipy]` of pip) |

The `url` of the recipes is the tarball placed in the GitHub Release (`https://github.com/parcel-jaea/PARCEL-release/releases/download/v1.2.0/parcel_sycl-1.2.0.tar.gz`).
Once it is placed in the Release, it is fetched and verified against the sha256 written in the recipes ("Installing from the Release Tarball" below). To use local sources,
specify the local package (the parent of this directory) with `spack develop`.

```bash
spack -e <env> repo add <package>/spack/spack_repo/parcel
spack -e <env> config add "concretizer:unify:when_possible"          # when installing CPU and GPU versions etc. side by side
spack -e <env> develop --no-clone --path <package> parcel-sycl@1.2.0
spack -e <env> develop --no-clone --path <package> py-parcel-sycl@1.2.0
spack -e <env> add "parcel-sycl@1.2.0 ~mpi ~cuda ~rocm" "py-parcel-sycl@1.2.0 ~mpi ~cuda ~rocm"                    # CPU
spack -e <env> add "parcel-sycl@1.2.0 ~mpi +cuda cuda_arch=70" "py-parcel-sycl@1.2.0 ~mpi +cuda cuda_arch=70"     # GPU
spack -e <env> concretize -f && spack -e <env> install

spack env activate <env>
spack load parcel-sycl+cuda py-parcel-sycl+cuda    # load by specifying the configuration
python script.py                                   # Python
acpp $PARCEL_CXXFLAGS -O3 -o prog prog.cpp $PARCEL_LDFLAGS        # C++
P=$(spack location -i parcel-sycl+cuda)            # C: spack load does not set the header and library locations, so give them yourself
gcc -I$P/include -o prog prog.c -L$P/lib -lparcel_sycl_c -Wl,-rpath,$P/lib   # (in the headers of the version without MPI, PARCEL_MPI defaults to 0)
```

- `spack load parcel-sycl` sets `PARCEL_CXXFLAGS` (target, optimization, with or without MPI, headers) and `PARCEL_LDFLAGS`
  (RPATHs of the libraries needed at run time (LLVM libomp.so, AdaptiveCpp, LAPACK, MPI) and LAPACK), for compiling user programs
  with the same settings as this package. With MPI, use
  `mpicxx $PARCEL_CXXFLAGS ... $PARCEL_LDFLAGS` (`spack load` also sets `OMPI_CXX=acpp` etc.).
  Neither calling `acpp` directly nor going through the MPI wrapper adds the RPATH of LLVM libomp.so, so add `PARCEL_LDFLAGS`.
- Specify root specs explicitly down to `~cuda ~rocm`. With only `parcel-sycl ~mpi`, unify concretization merges it with the GPU version.
- Installing CPU and GPU versions in the same environment makes them collide in the view, so exclude `parcel-sycl` and `py-parcel-sycl` from the environment view
  (`view: default: {root: .spack-env/view, exclude: [parcel-sycl, py-parcel-sycl]}`).
- Adding specs to an already concretized environment can make `concretize` fail with an internal error (unsatisfiable). Re-concretize with `concretize -f`.
- Use a single MPI and a single hipsycl in an environment (`packages:mpi:require:` down to the version and configuration, `packages:hipsycl:require:`).
  If several external MPIs are registered, another configuration (such as a CUDA version) can be selected.
- For the CUDA used by AdaptiveCpp, the `ACPP_CUDA_PATH` at build time (or spack's cuda) is recorded in `share/parcel/acpp_cuda_path`,
  and `spack load` sets the same value. To avoid using a CUDA newer than the driver, specify it with the external package cuda or this variable.
- The PARCEL headers are copied to `include/` and the LAPACKE headers to `include/lapacke/` (for environments whose compute nodes have no `lapacke.h`).
- **Post-install tests** `spack test run parcel-sycl py-parcel-sycl`: build and run the Quick Start with the installed headers and libraries
  (`parcel-sycl`: C++, and also C and Fortran with `+capi`; `py-parcel-sycl`: `quick_start.py`), and judge whether the maximum relative error
  against the exact solution is 1e-6 or less (the same judgment as `examples/run_quick_checks.sh`). Results with `spack test results`.
  Versions with MPI are launched with `MPIEXEC` (default: `bin/mpiexec` of the MPI) `MPIEXEC_NP_FLAG` (`-n`) `MPI_NP` (2) `MPIEXEC_ARGS`.
  GPU versions only test the build on nodes without a GPU (login nodes), and the run is SKIPPED, so also run them on GPU compute nodes.
  `PARCEL_TEST_RUN=0` stops the runs. The C and Fortran of `+fj` are not tested (use `check_stack.sh`).
  The number of threads and the decision on where it runs are the same as in `examples/run_quick_checks.sh`: if `OMP_NUM_THREADS` is not set,
  all available logical CPUs on a compute node (inside a job) and 4 on a login node (with MPI, divided by the number of processes so that the total stays the same).
  Whether it is a compute node is decided by the presence of `PBS_JOBID`, `PJM_JOBID` or `SLURM_JOB_ID`, and can be forced with `QUICK_NODE=compute|login`.
  The decision and the values used appear in the output of `spack test results -l`. For `+fj +mpi` (Fujitsu mpiexec), `RC_ONLY=1` is the default
  (the standard output of the ranks goes to the job log, where the check lines cannot be read, so only the exit status is judged; use `RC_ONLY=0` to turn it off).
  If the MPI is Open MPI and `MPIEXEC` is not given, `--bind-to none` is added unless `MPIEXEC_ARGS` specifies a placement. With Intel MPI, the C and Fortran
  tests are built with the generic wrappers in the same place (`mpicc`, `mpif90`), and the underlying compiler is set to spack's compiler (gfortran, etc.)
  with `I_MPI_CC` and `I_MPI_F90` (and `I_MPI_FC`) (the `mpiifx` returned by spack calls ifx, which cannot read `parcel_sycl_mod.mod` built by gfortran,
  and `mpiifx` with gfortran inside still points to the `mpi.mod` for ifx). Run the GPU versions and `+fj` (A64FX) on compute nodes
  (a short batch job or an interactive job). The CPU versions may be run on a login node (follow the rules of the site).
- Running tests inside the `spack develop` sources (which creates `bin/`, `__pycache__`, etc.) makes spack consider them updated,
  and the next `install` rebuilds PARCEL. Run tests on a copy of the sources.

### Installing from the Release Tarball

The recipes contain the sha256 of the tarball (`parcel_sycl-1.2.0.tar.gz`) made with `tools/make_tarball.py` (`--pin`;
`version("1.2.0", sha256="...")`). Once it is placed in the GitHub Release, it can be fetched from `url` and installed without `spack develop`
(`parcel-sycl` and `py-parcel-sycl` use the same tarball). spack verifies the fetched tarball against the sha256 in the recipes (stops if it differs).

Before it is placed in the Release, or when the machine cannot connect to the outside, place the tarball locally in the form of a spack mirror
(`_source-cache/archive/<first 2 characters of sha256>/<sha256>.tar.gz`).

```bash
python3 tools/make_tarball.py --outdir <outdir> --pin        # tarball and sha256; writes the sha256 into the 2 recipes
M=<mirror-dir>
SHA=$(cut -d' ' -f1 <outdir>/parcel_sycl-1.2.0.tar.gz.sha256)
mkdir -p $M/_source-cache/archive/${SHA:0:2}
cp <outdir>/parcel_sycl-1.2.0.tar.gz $M/_source-cache/archive/${SHA:0:2}/$SHA.tar.gz
spack -e <env> mirror add --scope env:<env> parcel-local file://$M
spack -e <env> repo add <package>/spack/spack_repo/parcel        # recipes with the sha256 written
spack -e <env> add "parcel-sycl@1.2.0 ~mpi ~cuda ~rocm"              # no develop
spack -e <env> concretize -f && spack -e <env> install
```

- The recipes inside the tarball are in the form without sha256 (`version("1.2.0")`) (because a recipe cannot contain the sha256 of the tarball itself).
  Remaking the tarball after writing the sha256 into the local recipes with `--pin` gives the same tarball. `--check` verifies it.
- When the package contents change, remake the tarball and rewrite the sha256 with `--pin`. For publication, use the recipes containing the sha256 of the published tarball.

### Choosing a Configuration

The configuration to install is chosen by the combination of package and variants (spec) (the version `@1.2.0` is omitted).

| Purpose | Spec |
|---|---|
| CPU, without MPI (C++, C, Fortran) | `parcel-sycl ~mpi ~cuda ~rocm` |
| CPU, with MPI | `parcel-sycl +mpi ~cuda ~rocm` |
| NVIDIA GPU (A100 as an example) | `parcel-sycl ~mpi +cuda cuda_arch=80` |
| NVIDIA GPU, with MPI | `parcel-sycl +mpi +cuda cuda_arch=80` |
| AMD GPU (MI300A) | `parcel-sycl ~mpi +rocm amdgpu_target=gfx942` |
| C++ (headers) only; no C / Fortran APIs | `parcel-sycl ~capi ~mpi ~cuda ~rocm` |
| Also use Python | Add `py-parcel-sycl` of the same configuration (e.g. `py-parcel-sycl ~mpi ~cuda ~rocm`) |

- C++ (headers) is always included in `parcel-sycl`. The C and Fortran APIs are included or excluded together with `capi` (default on) (one alone cannot be chosen).
  Python is chosen by whether `py-parcel-sycl` is added.
- `+capi` needs a Fortran compiler. In environments without one, use `~capi`.
- `py-parcel-sycl` uses `parcel-sycl` for its build, so even for Python only, `parcel-sycl` (with the default `+capi`) is installed. In environments without a Fortran compiler,
  add it as in `py-parcel-sycl ~mpi ~cuda ~rocm ^parcel-sycl ~capi` (confirmed on s86 that it concretizes and that `parcel-sycl` no longer uses a Fortran compiler).
  OpenBLAS also uses a Fortran compiler by default, so in environments with none at all, also specify `openblas ~fortran` (same as `envs/wis_ody`).
- `+cuda` and `+rocm` cannot be specified together. With neither, it is the CPU (OpenMP) version. For GPUs, specify one `cuda_arch` (V100 70, A100 80, GH200 90) or
  `amdgpu_target`.
- `+mpi` requires an MPI registered as an external package. Make `mpi`, `cuda`, `rocm`, etc. of `py-parcel-sycl` the same as the `parcel-sycl` it is combined with.
- The configurations installed by `stack/install_stack.sh` are those listed in `specs:` of `stack/envs/<env>/spack.yaml`. To narrow or change the configurations, rewrite `specs:` and then
  run `install_stack.sh` (if rewritten after installing, the same command installs only what is missing). Example (only C++, C, Fortran and Python for CPU with MPI):

```yaml
spack:
  specs:
  - parcel-sycl@1.2.0 +mpi ~cuda ~rocm
  - py-parcel-sycl@1.2.0 +mpi ~cuda ~rocm
```

## Installing from Scratch (`stack/`)

Without using an existing spack, a single script installs everything from fetching spack to PARCEL.

```bash
bash spack/stack/install_stack.sh s86 $HOME/parcel_stack 16           # <env> <prefix> [parallelism]
bash spack/stack/check_stack.sh $HOME/parcel_stack $HOME/parcel_check  # check on a login node
. $HOME/parcel_stack/setup.sh                                          # when using it
spack load parcel-sycl+mpi+cuda py-parcel-sycl+mpi+cuda
```

- Built by spack: LLVM 18.1.8, AdaptiveCpp (hipsycl 24.10.0), OpenBLAS, binutils, Python 3.11 with numpy, SciPy and pip,
  the PARCEL main body, the C / Fortran APIs and the Python package.
- Existing software of each environment (written as external packages in `stack/envs/<env>/spack.yaml`): compilers, CUDA / ROCm, MPI.
  These are tied to the network and drivers of the machine.
- spack itself (1.1.0.dev0) and the recipe collection spack-packages are pinned to the confirmed versions (at the top of `install_stack.sh`).
- spack, its settings and caches are all placed inside the prefix, and `SPACK_DISABLE_LOCAL_CONFIG=1` keeps `~/.spack` and `/etc/spack` from being read
  (neither affected by nor affecting existing spack environments).
- The PARCEL sources are the directory two levels above `install_stack.sh` (this package; it must have `Include/`, `capi/` and `python/`).
  To use another location, give `PARCEL_SRC=<dir>`. Keep it separate from the directory where tests are run (the note above).
- `envs/<env>/site_env.sh` is read during installation and when `setup.sh` is read (site modules for MPI etc.).
  `after_activate.sh` is read after the environment is activated, and `compilers.yaml` gives a compiler installed before the main body (environments whose OS GCC is old).
- If it stops midway, the same command resumes it. With `CONCRETIZE_ONLY=1`, only concretization is done.
- `check_stack.sh` runs `spack load` for each installed configuration and checks whether it can be imported from Python, and whether `sample_c.c`, `sample_f.f90` and
  `quick_start.cpp` of `examples/` build and all libraries needed for running are found (they are not run).
  Fortran without MPI is compiled together with `examples/sample_f_mpi_stub.f90`.
- Examples of settings for running regression tests and performance tests in the installed environment are `s86spack_cpu` / `s86spack_gpu` in `tests/site_config.py`
  (the "Running in a spack Environment" section of `tests/regression/README.md`).

### Provided Environments

| `envs/` | Machine | Configurations installed | Existing software | Confirmation |
|---|---|---|---|---|
| `s86` | x86_64 (Cascade Lake) + V100 | CPU / GPU (`cuda_arch=70`) × without / with MPI | GCC 9.5.0, CUDA 12.6, Open MPI 4.1.6 | `install_stack.sh` and `check_stack.sh` passed. Python, C, Fortran and C++ samples passed on compute nodes (with the version before the last changes to the recipes (V100 flags, Fortran module)) |
| `miyc` | Miyabi-C (Xeon MAX 9480) | CPU × without / with MPI | GCC 11.4.1, Open MPI 4.1.6 | `install_stack.sh` and `check_stack.sh` passed |
| `miyg` | Miyabi-G (GH200) | GPU (`cuda_arch=90`) × without / with MPI | GCC 11.4.1, CUDA 12.6, Open MPI 4.1.6 (CUDA version) | Same as above |
| `psa` | Plasma Simulator PS-A (Xeon 6980P) | CPU × without / with MPI (`ks_ortho=mgs`: the default orthogonalization of `ks_arnoldi` is MGS2) | GCC 11.5.0, Intel MPI 2021.18 | Same as above |
| `psb` | Plasma Simulator PS-B (MI300A) | GPU (`amdgpu_target=gfx942`) × without / with MPI | GCC 11.5.0, ROCm 6.3.3, Open MPI 5.0.7 (ROCm version) | Same as above |
| `wis_aqu` | Wisteria-Aquarius (A100) | GPU (`cuda_arch=80`) × without / with MPI | GCC 8.3.1 (only for building GCC 12), CUDA 12.2, Open MPI 4.1.1 | Same as above |
| `wis_ody` | Wisteria-Odyssey (A64FX) | CPU (`+fj`) × without / with MPI | GCC 8.3.1 (LLVM, boost, etc. and gfortran), Fujitsu compilers, MPI and LAPACK (tcsds-1.2.43) | `install_stack.sh` and `check_stack.sh` passed. Python, C, Fortran and C++ samples passed on compute nodes without and with MPI (np=1, 2, 4) |
| `generic` | Linux machines other than the 7 environments above (template) | CPU × without MPI (default). With MPI and NVIDIA GPUs are added by uncommenting | Rewrite before use (GCC, MPI, CUDA) | Confirmed by installing CPU and GPU × without and with MPI on Ubuntu 24.04 (WSL2, GCC 13.3.0, Open MPI 4.1.6, CUDA 12.6, RTX A6000) and Rocky Linux 9.8 (WSL2, GCC 11.5.0, Open MPI 4.1.1, CUDA 12.6, RTX A6000) |

- The locations of external packages (`prefix:`) and module names are those of each machine's system. Rewrite them for other machines.
- Wisteria-Odyssey (`wis_ody`) has x86 login nodes, so install it in a job on A64FX compute nodes (`envs/wis_ody/install_job.sh` is an example).
  LLVM 18 is built with the OS GCC 8.3.1, and its clang builds AdaptiveCpp and the C / C++ of numpy and SciPy (if GCC 12 or the like is built, the Fujitsu FCC
  cannot read its newer libstdc++ and C++ programs cannot be linked). The host side of PARCEL uses FCC, MPI uses `mpiFCC`, and LAPACK uses the Fujitsu
  `-lfjlapack`. OpenBLAS (for numpy and SciPy) is without Fortran, and numpy is pinned to 2.2 to match SciPy 1.13 (because GCC is older than 9.1).
  The parallelism is `PARCEL_STACK_JOBS` in `site_env.sh` (12; at 32 or more, the LLVM build runs out of memory). The LLVM build takes about 3 hours.
  When building user programs with the Fujitsu frt / fcc, add `-Kopenmp` (`after_activate.sh` sets it in `FFLAGS` / `CFLAGS`).
- On Linux machines other than the 7 environments above, copy and rewrite the template `envs/generic/` (e.g. `cp -r spack/stack/envs/generic spack/stack/envs/mysite`).
  The comments at the top of `spack.yaml` describe what to rewrite and how: the version and location of the compiler (GCC); when using MPI, its version and location;
  when using NVIDIA GPUs, the version and location of CUDA and `cuda_arch`; for CPUs other than x86_64, the `targets` of LLVM.
  On machines that load MPI etc. with modules, write `module load` in `site_env.sh`.
  The OS Open MPI of RHEL-family OSes (`dnf install openmpi-devel`) is loaded with `module load mpi/openmpi-x86_64`, and its headers are in `/usr/include/openmpi-x86_64`,
  not under `/usr/lib64/openmpi`. Using that directly as `prefix` makes the installation of PARCEL with MPI stop with
  `Query of package 'openmpi' for 'headers' failed`, so place `bin`, `lib`, `share` and `include` side by side in one directory
  (symbolic links) and use it as `prefix` (the MPI note in `spack.yaml`).
  Try only concretization with `CONCRETIZE_ONLY=1 bash spack/stack/install_stack.sh mysite <prefix>` before installing.
  This template was made after the included `s86` and `miyc` definitions, and installation has not been tried with it as-is on any machine.
  For AMD GPUs, base it on the `psb` definition (external registration of ROCm).
- macOS does not use spack but installs with Homebrew (`brew install adaptivecpp open-mpi openblas gcc numpy scipy`;
  for the Python package, source `python/envs/mac_brew.sh` and then `pip install`; for regression tests, `mac` in `tests/site_config.py`).
