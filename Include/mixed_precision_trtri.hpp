// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター


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
  // まずは work 配列に書き出して最後に戻す
  std::vector<double> Ainv(N*lda,0.0);
  for(int i = 0;i<N*lda;i++){
    Ainv[i] = 0.0;
  }
  for(int i = 0;i<N;i++){
    Ainv[i+i*lda] = 1.0;
  }

  if(uplo == "U"){
    for(int i = N-1;i>=0;i--){
      T_calc Aii = A[i+i*lda];
      for(int j = i;j<N;j++){
	for(int k = 0;k<i;k++){
	  T_calc Aki  = A[k+i*lda];
	  T_calc Aval = Aki/Aii;
	  Ainv[k+j*lda] = Ainv[k+j*lda] - Ainv[i+j*lda]*Aval;
	}
      }
      for(int j = i;j<N;j++){
	for(int k = 0;k<i;k++){
	  T_calc Aki  = A[k+i*lda];
	  T_calc Aval = Aki/Aii;
	  A[k+j*lda]  = A[k+j*lda] - A[i+j*lda]*Aval;
	}
	A[i+j*lda] =    A[i+j*lda] / Aii;
      	Ainv[i+j*lda] = Ainv[i+j*lda] / Aii;
      }


    }

    for(int j = 0;j<N;j++){
      int i = 0;
      T_calc Aii = A[i+i*lda];
         A[i+j*lda] =    A[i+j*lda] / Aii;
      Ainv[i+j*lda] = Ainv[i+j*lda] / Aii;
    }

  }



  for(int i = 0;i<N*lda;i++){
    A[i] = Ainv[i];
  }


  return ;

}


#endif
