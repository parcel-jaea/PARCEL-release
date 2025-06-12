#include <iostream>
// #include "vec.hpp"
#include "poisson.hpp"
#include "mat.hpp"
#include "pcg.hpp"
#include "pgmres.hpp"

#define NX 10
#define problem_size NX*NX*NX
//#define problem_type long double
#define problem_type double

int main(int argc, char* argv[]){

#if 1
  // CRS
  poisson_equation<problem_type> poisson;
  PARCEL_INT nnz = poisson.get_nnz(NX);
  crs_mat<problem_type> crsMatrix(problem_size,nnz);
  poisson.make_matrix(NX,crsMatrix);

  std::vector<problem_type> x(problem_size,0.0);
  std::vector<problem_type> b(problem_size,0.0);
  poisson.make_right_vector(NX,b);
  for(int i=0;i<b.size();i++){
    b[i] = 1.0;
  }
#if 0
  pcg<problem_type> pcg_solver;
  pcg_solver.set_residualnorm(1.0e-10);
  pcg_solver.set_itrmax(20);
  // std::cout<<x[0]<<std::endl;
  // std::cout<<x[1]<<std::endl;
  // std::cout<<x[2]<<std::endl;
  // std::cout<<x[3]<<std::endl;
  // std::cout<<x[4]<<std::endl;
  pcg_solver.solve(crsMatrix,x,b);
  poisson.check_solution(x,NX);
  // std::cout<<x[0]<<std::endl;
  // std::cout<<x[1]<<std::endl;
  // std::cout<<x[2]<<std::endl;
  // std::cout<<x[3]<<std::endl;
  // std::cout<<x[4]<<std::endl;
#else
  pgmres<problem_type> pgmres_solver;
  pgmres_solver.set_residualnorm(1.0e-10);
  pgmres_solver.set_itrmax(4);
  pgmres_solver.solve(crsMatrix,x,b);
  poisson.check_solution(x,NX);
#endif
  
#endif

#if 0
  // DIA
  poisson_equation<problem_type> poisson;
  PARCEL_INT nnd = poisson.get_nnd();
  dia_mat<problem_type> diaMatrix(problem_size,nnd);
  poisson.make_matrix(NX,diaMatrix);

  std::vector<problem_type> x(problem_size,0.0);
  std::vector<problem_type> b(problem_size,0.0);
  poisson.make_right_vector(NX,b);

  pcg<problem_type> pcg_solver;
  pcg_solver.set_residualnorm(1.0e-10);
  pcg_solver.set_itrmax(100);
  pcg_solver.solve(diaMatrix,x,b);
  poisson.check_solution(x,NX);

#endif

  return 0;

}
