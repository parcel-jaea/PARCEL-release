English | [日本語](SUPPORT.ja.md)

# Support

## Contact

Send bug reports and questions to GitHub Issues of the public repository.

- Contact: <https://github.com/parcel-jaea/PARCEL-release/issues>
- Distribution site: <https://ccse2.jaea.go.jp/software/parcel/>
- Public repository: <https://github.com/parcel-jaea/PARCEL-release> (the release tarball is placed in Releases)

## What to Include in a Report

Please attach the following to bug reports and questions.

- The version of PARCEL_SYCL (`parcel::version_string()` for C++, `parcel_version_string()` for C / Fortran,
  `parcel.__version__` and `parcel.library_version()` for Python. Example: `1.2.0`) and the name and sha256 of the release tarball (if known)
- How it was installed (built directly from source / spack `parcel-sycl` / `py-parcel-sycl` / pip) and the build commands
  (`--acpp-targets` and the macros given with `-D`. For spack, the output of `spack spec` and `share/parcel/build_config.txt`)
- The computer and OS, and the CPU / GPU model
- The SYCL implementation (the AdaptiveCpp version and the version of the LLVM it was built with), the host compiler, the CUDA / ROCm version, and the MPI implementation and version
- The solver, preconditioner and precision settings used, the matrix size (number of rows and nonzeros), and the number of ranks and threads
- Steps to reproduce (preferably a minimal example that reproduces the problem with a verification sample in `examples/` or `tests/verify/`), and
  the standard output of rank 0 (including the ` converged_reason = ...` line) and the error messages

For how to run the tests, see `tests/regression/README.md` (regression tests).
