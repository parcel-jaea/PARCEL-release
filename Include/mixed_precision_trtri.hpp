

#ifndef MIXED_PRECISON_TRTRI_HPP_

#define MIXED_PRECISON_TRTRI_HPP_

// 三角行列の逆行列を求める.
// LAPACK準拠 (Column - major order)
// UPLO is CHARACTER*1
//   'U':  A is upper triangular;
//   'L':  A is lower triangular.

// DIAG is CHARACTER*1
//   'N':  A is non-unit triangular;
//   'U':  A is unit triangular.

// N is INTEGER
//   The order of the matrix A.  N >= 0.

// A is DOUBLE PRECISION array, dimension (LDA,N)
//   On entry, the triangular matrix A.  If UPLO = 'U', the
//   leading N-by-N upper triangular part of the array A contains
//   the upper triangular matrix, and the strictly lower
//   triangular part of A is not referenced.  If UPLO = 'L', the
//   leading N-by-N lower triangular part of the array A contains
//   the lower triangular matrix, and the strictly upper
//   triangular part of A is not referenced.  If DIAG = 'U', the
//   diagonal elements of A are also not referenced and are
//   assumed to be 1.
//   On exit, the (triangular) inverse of the original matrix, in
//   the same storage format.

// LDA is INTEGER
//   The leading dimension of the array A.  LDA >= max(1,N).
//   INFO is INTEGER
//   = 0: successful exit
//   < 0: if INFO = -i, the i-th argument had an illegal value
//   > 0: if INFO = i, A(i,i) is exactly zero.  The triangular
//               matrix is singular and its inverse can not be computed.

#ifndef PARCEL_INT

#define PARCEL_INT int 

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

#endif

template<
  class T_mat,
  class T_calc
  >
void trtri(
	   std::string &uplo,
	   std::string &diag,
	   T_mat *A,
	   PARCEL_INT N,
	   PARCEL_INT lda
	   ){
  // 単純な掃き出し法で実装
  // 上三角行列のみ対応
  // まずはwork配列に書き出して最後に戻す 
  std::vector<double> Ainv(N*lda,0.0);
  for(int i = 0;i<N*lda;i++){
    Ainv[i] = 0.0;
  }
  for(int i = 0;i<N;i++){
    Ainv[i+i*lda] = 1.0;
  }

  if(uplo == "U"){
    //    std::cout<<"Upper Triangular Matrix"<<std::endl;
    for(int i = N-1;i>=0;i--){
      T_calc Aii = A[i+i*lda];
      for(int j = i;j<N;j++){
	// std::cout<<" ----- "<<std::endl;
	// std::cout<<i<<","<<j<<" : "<<0<<" --> "<<i-1<<std::endl;
	for(int k = 0;k<i;k++){
	  T_calc Aki  = A[k+i*lda];
	  T_calc Aval = Aki/Aii;
	  // std::cout<<"     k = "<<k<<" : Aval = "<<Aval<<std::endl;
	  Ainv[k+j*lda] = Ainv[k+j*lda] - Ainv[i+j*lda]*Aval;
	}
      }
      for(int j = i;j<N;j++){
	for(int k = 0;k<i;k++){
	  // std::cout<<k<<","<<j<<std::endl;
	  T_calc Aki  = A[k+i*lda];
	  T_calc Aval = Aki/Aii;
	  A[k+j*lda]  = A[k+j*lda] - A[i+j*lda]*Aval;
	}
	A[i+j*lda] =    A[i+j*lda] / Aii;
      	Ainv[i+j*lda] = Ainv[i+j*lda] / Aii;
	// std::cout<<" ----- "<<std::endl;
      }

#if 0
      for(int jj = 0;jj<N;jj++){
	for(int ii = 0;ii<N;ii++){
	  std::cout<<A[jj+ii*N]<<", ";
	}
	std::cout<<" | ";
	for(int ii = 0;ii<N;ii++){
	  std::cout<<Ainv[jj+ii*N]<<", ";
	}
	std::cout<<std::endl;
      }
      std::cout<<" ------------------- "<<std::endl<<std::endl;
#endif

    }

    for(int j = 0;j<N;j++){
      int i = 0;
      T_calc Aii = A[i+i*lda];
         A[i+j*lda] =    A[i+j*lda] / Aii;
      Ainv[i+j*lda] = Ainv[i+j*lda] / Aii;
    }

  }


#if 0
  // 元の行列が単位行列になることを確認
  for(int jj = 0;jj<N;jj++){
    for(int ii = 0;ii<N;ii++){
      std::cout<<A[jj+ii*N]<<", ";
    }
    std::cout<<" | ";
    for(int ii = 0;ii<N;ii++){
      std::cout<<Ainv[jj+ii*N]<<", ";
    }
    std::cout<<std::endl;
  }
#endif

#if 1
  for(int i = 0;i<N*lda;i++){
    A[i] = Ainv[i];
  }
#endif

#if 0
  if(uplo == "L"){
    std::cout<<"Lower Triangular Matrix"<<std::endl;
  }
  if(diag == "N"){
    std::cout<<"Diag is non unit"<<std::endl;
  }
  if(diag == "U"){
    std::cout<<"Diag is unit"<<std::endl;
  }
#endif

  return ;

}

#if 0

// 単体テスト
int main(){
  const int N = 8;
  std::vector<double> A(N*N,0.0);
  std::vector<double> Ainv(N*N,0.0);
  std::vector<double> I(N*N,0.0);

  for(int i = 0;i<N*N;i++){
    A[i] = 0.0;
    Ainv[i] = 0.0;
  }
#if 0
  //  下三角行列
  for(int i = 0;i<N;i++){
    for(int j = i;j<N;j++){
      A[j+i*N] = i+j;
    }
  }
#endif

#if 1
  //  上三角行列
  for(int i = 0;i<N;i++){
    for(int j = 0;j<=i;j++){
      A[j+i*N] = (i+1)+(j+1);
      //      A[j+i*N] = (i+1)+(j+1) + (double)rand() / (double)RAND_MAX;
    }
  }
#endif
  for(int i = 0;i<N*N;i++){
    Ainv[i] = A[i];
  }
  std::string uplo("U");
  std::string diag("N");
  trtri<double,double>(uplo,diag,Ainv.data(),N,N);

  for(int i = 0;i<N;i++){
    for(int j = 0;j<N;j++){
      double tmp = 0.0;
      for(int k = 0;k<N;k++){
	tmp = tmp + A[j+k*N] * Ainv[k+i*N];
      }
      I[j+i*N] = tmp;
    }
  }

  std::cout<<"Matrix "<<std::endl;
  for(int j = 0;j<N;j++){
    for(int i = 0;i<N;i++){
      std::cout<<A[j+i*N]<<", ";
    }
    std::cout<<std::endl;
  }

  std::cout<<" ----------- "<<std::endl;
  std::cout<<"Inverse Matrix "<<std::endl;
  for(int j = 0;j<N;j++){
    for(int i = 0;i<N;i++){
      std::cout<<Ainv[j+i*N]<<", ";
    }
    std::cout<<std::endl;
  }

  std::cout<<" ----------- "<<std::endl;
  std::cout<<" Test : Unit Matrix  "<<std::endl;
  for(int j = 0;j<N;j++){
    for(int i = 0;i<N;i++){
      std::cout<<I[j+i*N]<<", ";
    }
    std::cout<<std::endl;
  }


  return 0;

}

#endif

#endif
