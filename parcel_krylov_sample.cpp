#include "./Include/parcel_sycl.hpp"
#include "./parcel_default.hpp"
#include "./poisson_cpu.hpp"

int main(int argc, char *argv[]) {
  
  sycl::device device;
  sycl::property::queue::in_order property;
#ifdef HIPSYCL_SYCL_HPP
  sycl::queue q;
#else
  sycl::queue q{device,property};
  q = sycl::default_selector{};
#endif
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

#if PARCEL_MPI  
  MPI_Init(&argc, &argv);
  parcel::comm comm(MPI_COMM_WORLD,&q);
  int myrank   = comm.mpi_rank();
  int nprocess = comm.mpi_size();
  PARCEL_INT nlocal = problem_size / nprocess;
  PARCEL_INT nstart = nlocal*myrank;
  PARCEL_INT nend   = nlocal*(myrank+1)-1;
  if( myrank == (nprocess-1) ){ nend = problem_size-1; }
#else
  int myrank   = 0;
  int nprocess = 1;
  PARCEL_INT nlocal = problem_size;
  PARCEL_INT nstart = 0;
  PARCEL_INT nend   = problem_size-1;
#endif
  
  double tolerance = 1.0e-9;
  int itrmax = MAX_ITERATION;

  // 3D poisson matrix 
  poisson_equation<problem_type> poisson;
  auto nnz = poisson.get_nnz(NX);
  crs_mat<problem_type> crsMatrix_cpu(problem_size,nnz);
  poisson.make_matrix(NX,crsMatrix_cpu);

#if PARCEL_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  parcel::matrix<parcel_solver_data_type> parcel_matrix(&q);
  PARCEL_INT nblock = nend - nstart  + 1;
  PARCEL_INT *crsRowptr_cpu = crsMatrix_cpu.crsRowptr_data();
  PARCEL_INT *crsCol_cpu = crsMatrix_cpu.crsCol_data();
  problem_type *crsA_cpu = crsMatrix_cpu.A_data();
  PARCEL_INT nnz_local = 0;
  for(int irow = nstart;irow<=nend;irow++){
    for(int i = crsRowptr_cpu[irow];i<crsRowptr_cpu[irow+1];i++){
      nnz_local = nnz_local + 1;
    }
  }
  std::vector< problem_type > crsA_local(nnz_local);
  std::vector< PARCEL_INT >   crsCol_local(nnz_local);
  std::vector< PARCEL_INT >   crsRowptr_local(nblock+1);
  
  crsRowptr_local[0] = 0;
  PARCEL_INT nnz_ptr = 0;
  for(int irow = nstart;irow<=nend;irow++){
    crsRowptr_local[irow-nstart+1] = crsRowptr_local[irow-nstart] + crsRowptr_cpu[irow+1] - crsRowptr_cpu[irow];
    for(int i = crsRowptr_cpu[irow];i<crsRowptr_cpu[irow+1];i++){
      crsA_local[nnz_ptr]   = crsA_cpu[i];
      crsCol_local[nnz_ptr] = crsCol_cpu[i];      
      nnz_ptr = nnz_ptr + 1;
    }
  }
  
  parcel_matrix.set_crsMat<problem_type>(
#if PARCEL_MPI
					 &comm,nstart,nend,
					 nblock,nnz_local,
					 crsRowptr_local.data(),
					 crsCol_local.data(),
					 crsA_local.data()
#else
					 problem_size,nnz,					 
					 crsRowptr_local.data(),
					 crsCol_local.data(),
					 crsA_local.data()
#endif
					    );
  crsMatrix_cpu.reset();

  parcel::vector<parcel_solver_data_type> vecx(nblock,&q);
  parcel::vector<parcel_solver_data_type> vecy(nblock,&q);
  parcel::vector<parcel_solver_data_type> vecb(nblock,&q);
  std::vector<problem_type> x_cpu_local(nblock);
  std::vector<problem_type> b_cpu_local(nblock);  
  for(int irow=nstart;irow<=nend;irow++){
    x_cpu_local[irow-nstart] = 0.0;
    b_cpu_local[irow-nstart] = 1.0;
  }  
  vecx.setValue(x_cpu_local.data());
  vecb.setValue(b_cpu_local.data());

#if pcg_solver
  parcel::krylov::pcg<
    parcel_solver_calc_type, // Tspmv
    parcel_solver_calc_precon_type, // Tprecon
    parcel_solver_calc_type, // Taxpy
    parcel_solver_calc_type  // dot
    > solver;
#endif
#if pbicgstab_solver
  parcel::krylov::pbicgstab<
    parcel_solver_calc_type, // Tspmv
    parcel_solver_calc_precon_type, // Tprecon
    parcel_solver_calc_type, // Taxpy
    parcel_solver_calc_type  // dot
    > solver;
#endif
#if pgmres_solver
  parcel::krylov::pgmres<
    parcel_solver_calc_type, // Tspmv
    parcel_solver_calc_precon_type, // Tprecon
    parcel_solver_calc_type, // Taxpy
    parcel_solver_calc_type  // dot
    > solver;
#endif
#if pcagmres_solver
    parcel::krylov::pcagmres<
    parcel_solver_calc_type, // Tspmv
    parcel_solver_calc_precon_type, // Tprecon
    parcel_solver_calc_type, // Taxpy
    parcel_solver_calc_type  // dot
    > solver;
#endif
  
#if pgmres_solver
    solver.set_restart(GMRES_RESTART);
    solver.set_qr_algo(GMRES_QR_ALGO);
#endif

  solver.set_tolerance(tolerance);
  solver.set_max_iteration(itrmax);

  // 前処理設定
  PARCEL_INT precon_algo = parcel_solver_precon_algo;
  switch (precon_algo) {
  case parcel::PRECON_NONE:
    // none precon
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

  for(int i = 0;i<1;i++){  
    vecx.setValue(x_cpu_local.data());
    q.wait();
    solver.solver<
      parcel_solver_data_type, // Tmat  :: 係数行列データ精度
      parcel_solver_data_type, // Tvecx :: 入力ベクトルデータ精度 
      parcel_solver_data_type, // Tvecb :: 右辺ベクトルデータ精度
      parcel_solver_data_type, // Tvec  :: クリロフ部分空間法内部データ精度
      parcel_solver_data_precon_mat_type, // Tprecon_mat :: 前処理行列データ精度
      parcel_solver_data_precon_vec_type // Tprecon_vec :: 前処ベクトルデータ精度
#if pgmres_solver
      ,parcel_solver_data_type, // グラムシュミットの直交化精度
      parcel_solver_data_type   // ヘッセンベルグ行列の精度
#endif      
#if pcagmres_solver
      ,parcel_solver_data_type, // 基底ベクトルの精度
      parcel_solver_calc_qr_type,
      parcel_solver_calc_qr_type
#endif    
    >(parcel_matrix,vecx,vecb);
  q.wait();
  }

  vecx.getValue(x_cpu_local.data());  q.wait();
  q.wait();
  if( myrank == 0){    
    for(int i = 0;i<(10>nblock?nblock:10);i++){
      std::cout<<"x["<<i<<"] = "<<x_cpu_local[i]<<std::endl;
    }
  }
  
#if PARCEL_MPI
  MPI_Finalize(); 
#endif
  
  return 0;

}
