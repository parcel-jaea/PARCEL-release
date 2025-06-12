#include <iostream>
#include <quadmath.h>

// #include "vec.hpp"
// #include "poisson.hpp"
#include "Toepliz.hpp"
#include "mat.hpp"
#include "pcg.hpp"

#define NX 5
#define problem_size NX*NX*NX
// #define problem_type long double
// #define problem_type double
#define problem_type long double

int main(int argc, char* argv[]){

  toepliz_equation<problem_type> toepliz;
  //  poisson_equation<problem_type> poisson;
  auto nnz = toepliz.get_nnz(problem_size);
  crs_mat<problem_type> crsMatrix(problem_size,nnz);
  std::vector<problem_type> x(problem_size,0.0);
  std::vector<problem_type> b(problem_size,0.0);
  double gamma = 2.0;
  toepliz.make_matrix(problem_size,gamma,crsMatrix);
  toepliz.make_right_vector(crsMatrix,b);
  pcg<problem_type> pcg_solver;
  pcg_solver.set_residualnorm(1.0e-10);
  pcg_solver.set_itrmax(200);
  pcg_solver.solve(crsMatrix,x,b);
  toepliz.check_solution(x);

}
