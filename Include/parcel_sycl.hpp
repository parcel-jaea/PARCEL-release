#ifndef PARCEL_SYCL_HPP_
#define PARCEL_SYCL_HPP_

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <vector>

#define PARCEL_MPI 1

#ifndef PARCEL_INT
#define PARCEL_INT int32_t
#define MPI_PARCEL_INT MPI_INT32_T
#else
#define PARCEL_INT int64_t
#define MPI_PARCEL_INT MPI_INT64_T
#endif

#include "ddlib.hpp"
#include "util_sycl.hpp"
#include "parcel.hpp"
#include "parcel_blas.hpp"
#include "parcel_krylov.hpp"

#if PARCEL_MPI
#include <mpi.h>
#include "./parcel_mpi/parcel_mpi.hpp"
#endif

#endif
