#ifndef PARCEL_CRS_HPP_
#define PARCEL_CRS_HPP_

#ifndef _BLOCKING_LOOP_
#define _BLOCKING_LOOP_ 0
#endif

#ifndef _A64FX_HIPSYCL_
#define _A64FX_HIPSYCL_ 0
#endif

#if _A64FX_HIPSYCL_

#define CRS_SPMV_NBLOCK 12
#define CRS_SPMV_BLOCKSIZE 4096
#else
#define CRS_SPMV_NBLOCK    N
#define CRS_SPMV_BLOCKSIZE 1
#endif

#endif
