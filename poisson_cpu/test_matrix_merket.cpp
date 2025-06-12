
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <vector>
#include <fstream>

// #define problem_type long double
#define problem_type double

#include "poisson.hpp"
#include "Toepliz.hpp"
#include "mat.hpp"
#include "pcg.hpp"
#include "pbicg.hpp"

#define pcg_solver 1
#define pbicgstab_solver 0

int main(int argc, char *argv[]) {

  double tolerance = 1.0e-10;
  int itrmax = 1000;

  // --------
  // コマンドライン引数のチェック
  if(argc != 2){
    std::cerr << "引数の数が間違っています．" << std::endl;
    std::cerr << "./extraction input.txt" << std::endl;
    return 1;
  }

  // C++のファイル入力のための準備
  std::ifstream ifs(argv[1], std::ios::in);

  // ファイルを開くのに失敗したときの処理
  if(!ifs){
    std::cerr << "Error: file not opened." << std::endl;
    return 1;
  }

  // string str;
  PARCEL_INT nrow,ncol,nnz;
  ifs >> nrow >> ncol >> nnz;

  std::cout<<"row = "<<nrow<<std::endl;
  std::cout<<"col = "<<ncol<<std::endl;
  std::cout<<"nnz = "<<nnz<<std::endl;

  std::vector<problem_type> coo_A(nnz,0.0);
  std::vector<PARCEL_INT> coo_row(nnz,0);
  std::vector<PARCEL_INT> coo_col(nnz,0);
  problem_type *cooA = coo_A.data();
  PARCEL_INT *coorow = coo_row.data();
  PARCEL_INT *coocol = coo_col.data();

  double matA;
  PARCEL_INT row,col;
  PARCEL_INT count = 0;
  // 抽出演算子>>を使ってデリミタで区切られた単語，値を読み込む
  while(ifs >> row >> col >> matA){ // str == "name:"; num == number;
    //    cout << str << " " << row << ","<< col <<","<< matA <<endl; // そのまま出力
    cooA[count]   = matA;
    coorow[count] = row-1; // 0始まりのため
    coocol[count] = col-1; // 0始まりのため
    count = count+1;
  }
  // ファイルを閉じる
  ifs.close();

  crs_mat<problem_type> crsMatrix_cpu(nrow,nnz);
  problem_type *A    = crsMatrix_cpu.A_data();
  PARCEL_INT *crsRowptr = crsMatrix_cpu.crsRowptr_data();
  PARCEL_INT *crsCol    = crsMatrix_cpu.crsCol_data();

  PARCEL_INT row0 = coorow[0];
  PARCEL_INT row_ptr_n=0;
  crsRowptr[ row_ptr_n ] = 0;
  for(PARCEL_INT i = 0;i<nnz;i++){
    if(row0 != coorow[i]){
      row0 = coorow[i];
      row_ptr_n = row_ptr_n + 1;
      crsRowptr[ row_ptr_n ] = i;
    }
    A[i]         = cooA[i];
    crsCol[i]    = coocol[i];
  }
  crsRowptr[nrow] = nnz;

  //  for(i=0;i<nrow;i++){
  for(PARCEL_INT i=0;i<5;i++){
    for(PARCEL_INT j=crsRowptr[i];j<crsRowptr[i+1];j++){
      //      std::cout<<" CRS : "<< i+1         << " , " << crsCol[j]+1 << " , " << A[j]    <<std::endl;
      //      std::cout<<" COO : "<< coorow[j] << " , " << coocol[j] << " , " << cooA[j] <<std::endl;
    }
  }
  for(PARCEL_INT i=nrow-5;i<nrow;i++){
    std::cout<<"row = "<<i<<" : "<<crsRowptr[i]<<" --> "<<crsRowptr[i+1]<<std::endl;
    for(PARCEL_INT j=crsRowptr[i];j<crsRowptr[i+1];j++){
      //      std::cout<<" CRS : "<< i+1         << " , " << crsCol[j]+1 << " , " << A[j]    <<std::endl;
      //      std::cout<<" COO : "<< coorow[j] << " , " << coocol[j] << " , " << cooA[j] <<std::endl;
    }
  }

  std::vector<problem_type> x_cpu(nrow,0.0);
  std::vector<problem_type> b_cpu(nrow,0.0);
  problem_type *xcpu = x_cpu.data();
  problem_type *bcpu = b_cpu.data();
  for(PARCEL_INT i = 0;i<nrow;i++){
    xcpu[i] = 0.0;
    bcpu[i] = 1.0;
  }
  ///  pbicg<problem_type> krylov_solver;

#if pcg_solver
  pcg<problem_type> krylov_solver;
#endif

#if pbicgstab_solver
  pbicg<problem_type> krylov_solver;
#endif

  // CPU
  krylov_solver.set_residualnorm(tolerance);
  krylov_solver.set_itrmax(itrmax);
  krylov_solver.solve(crsMatrix_cpu,x_cpu,b_cpu);

  return 0;
}
