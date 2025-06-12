#include <mpi.h>

#include "./Include/parcel_sycl.hpp"
#include "./parcel_default.hpp"

void run_parcel_sycl_main( 
			  MPI_Comm mpi_comm_c,
			  int n, int nnz,
			  double *crsA, 
			  int *crsCol, 
			  int *crsRow_ptr, 
			  double *vecx, 
			  double *vecb 
			   ) {

  sycl::device device;
  sycl::property::queue::in_order property;
#ifdef HIPSYCL_SYCL_HPP
  sycl::queue q;
#else
  sycl::queue q{device,property};
  q = sycl::default_selector{};
#endif

  parcel::comm comm(mpi_comm_c,&q);
  int myrank   = comm.mpi_rank();
  int nprocess = comm.mpi_size();

  if(myrank == 0)std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  parcel::matrix<parcel_solver_data_type> parcel_matrix(&q);
  PARCEL_INT nlocal = n;
  PARCEL_INT nstart = nlocal*myrank;
  PARCEL_INT nend   = nlocal*(myrank+1)-1;

  q.wait();
  parcel_matrix.set_crsMat<problem_type>(
					 &comm,nstart,nend,
					 n,nnz,
					 crsRow_ptr,
					 crsCol,
					 crsA
					 );
  q.wait();

  parcel::vector<parcel_solver_data_type> vx(n,&q);
  parcel::vector<parcel_solver_data_type> vb(n,&q);
  vx.setValue(vecx);
  vb.setValue(vecb);


  parcel::krylov::pcg<
    parcel_solver_calc_type,
    parcel_solver_calc_precon_type,
    parcel_solver_calc_type,
    parcel_solver_calc_type 
    > solver;
  solver.set_tolerance( 1.0e-9 );
  solver.set_max_iteration( MAX_ITERATION );

  PARCEL_INT precon_algo = parcel_solver_precon_algo;
  switch (precon_algo) {
  case parcel::PRECON_NONE:
    // none 
    solver.set_precon_algo(parcel::PRECON_NONE);
    break;
  case parcel::PRECON_POINTJACOBI:
    // point jacobi precon
    solver.set_precon_algo(parcel::PRECON_POINTJACOBI);
    break;
  case parcel::PRECON_BLOCKJACOBI:
    // block jacobi precon
    solver.set_precon_algo(parcel::PRECON_BLOCKJACOBI);
    solver.set_scaling_matrix(true);
    solver.set_scaling_vector(true);
    solver.set_blockjacobi(BLOCKJACOBI_BLOCK);
    break;
  case parcel::PRECON_JACOBI:
    // jacobi precon
    solver.set_precon_algo(parcel::PRECON_JACOBI);
    solver.set_scaling_matrix(true);
    solver.set_scaling_vector(true);
    solver.set_jacobi(JACOBI_ITER);
    break;
  }
  q.wait();
    
  solver.solver<
    parcel_solver_data_type,
    parcel_solver_data_type,
    parcel_solver_data_type,
    parcel_solver_data_type,
    parcel_solver_data_precon_mat_type,
    parcel_solver_data_precon_vec_type
    >(parcel_matrix,vx,vb);
  vx.getValue(vecx);  q.wait();
  q.wait();
  
  return ;
}
  

extern "C" {
  void run_parcel_sycl_f( 
			 int *mpi_comm_f, 
			 int n, int nnz,
			 double *crsA, 
			 int *crsCol, 
			 int *crsRow_ptr, 
			 double *vecx, 
			 double *vecb 
			  ) {
    MPI_Comm mpi_comm_c = MPI_Comm_f2c(mpi_comm_f[0]);
    run_parcel_sycl_main( 
			 mpi_comm_c, 
			 n, nnz,
			 crsA, 
			 crsCol, 
			 crsRow_ptr, 
			 vecx, 
			 vecb );
    return ;
  }
  void run_parcel_sycl_c( 
			 MPI_Comm mpi_comm_c, 
			 int n, int nnz,
			 double *crsA, 
			 int *crsCol, 
			 int *crsRow_ptr, 
			 double *vecx, 
			 double *vecb 
			  ) {
    run_parcel_sycl_main( 
			 mpi_comm_c, 
			 n, nnz,
			 crsA, 
			 crsCol, 
			 crsRow_ptr, 
			 vecx, 
			 vecb );
    return ;
  }
}
