#ifndef PARCEL_DEFAULT_HPP_
#define PARCEL_DEFAULT_HPP_

#ifndef MAX_ITERATION
#define MAX_ITERATION 10
#endif

#ifndef NX
#define NX 240
#endif

#ifndef GMRES_QR_ALGO
#define GMRES_QR_ALGO  parcel::qr::QR_MGS
#endif

#ifndef parcel_solver_calc_qr_type
#define parcel_solver_calc_qr_type parcel_double
#endif

#ifndef CAGMRES_RESTART_S
#define CAGMRES_RESTART_S 8
#endif
#ifndef CAGMRES_RESTART_T
#define CAGMRES_RESTART_T 1
#endif
#ifndef GMRES_RESTART
#define GMRES_RESTART 8
#endif

#ifndef parcel_solver_data_type
#define parcel_solver_data_type parcel_double
#endif

#ifndef parcel_solver_calc_type
#define parcel_solver_calc_type parcel_double
#endif

#ifndef parcel_solver_calc_precon_type
#define parcel_solver_calc_precon_type parcel_double
#endif

#ifndef parcel_solver_data_precon_vec_type
#define parcel_solver_data_precon_vec_type parcel_double
#endif

#ifndef parcel_solver_data_precon_mat_type
#define parcel_solver_data_precon_mat_type parcel_double
#endif

#ifndef parcel_solver_precon_algo
#define parcel_solver_precon_algo 0
#endif

#ifndef JACOBI_ITER
#define JACOBI_ITER 1
#endif

#ifndef BLOCKJACOBI_BLOCK
#define BLOCKJACOBI_BLOCK 1
#endif

#define problem_size (NX*NX*NX)
#define problem_type double

#define pcg_solver 1
// #define pbicgstab_solver 1
// #define pgmres_solver 1
// #define pcagmres_solver 1

#endif
