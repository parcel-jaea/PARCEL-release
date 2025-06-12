#ifndef PARCEL_PRECON_HPP_
#define PARCEL_PRECON_HPP_

#include <type_traits>
#include <algorithm>
#include "parcel.hpp"
#include "mixed_precision.hpp"
#include "parcel_util.hpp"
#include "parcel_type.hpp"
#include "parcel_precon_blockjacobi.hpp"
#include "parcel_precon_pointjacobi.hpp"
#include "parcel_precon_jacobi.hpp"

namespace parcel{  

  const PARCEL_INT PRECON_NONE        = 0;
  const PARCEL_INT PRECON_POINTJACOBI = 1;
  const PARCEL_INT PRECON_BLOCKJACOBI = 2;
  const PARCEL_INT PRECON_JACOBI      = 4;

  class precon_param{

    PARCEL_INT precon_algo_ = PRECON_NONE;
    PARCEL_INT JACOBI_ITER_ = 1;
    PARCEL_INT BLOCK_JACOBI_NUMBLOCK_ = 1;
    bool SCALING_MATRIX_ = false;
    bool SCALING_VECTOR_ = false;

  public :
    precon_param(){}
    ~precon_param(){}
    void set_precon_algo(PARCEL_INT precon_algo){
      precon_algo_ = precon_algo;
    }
    void set_jacobi(PARCEL_INT ITER){
      JACOBI_ITER_ = ITER;
    }
    void set_blockjacobi(PARCEL_INT BLOCK_JACOBI_NUMBLOCK){
      BLOCK_JACOBI_NUMBLOCK_ = BLOCK_JACOBI_NUMBLOCK;
    }

    void set_scaling_matrix( bool SCALING_MATRIX){
      SCALING_MATRIX_ = SCALING_MATRIX;
      return ;
    }

    void set_scaling_vector( bool SCALING_VECTOR){
      SCALING_VECTOR_ = SCALING_VECTOR;
      return ;
    }

    PARCEL_INT get_precon_algo(){
      return precon_algo_;
    }
    PARCEL_INT get_jacobi(){
      return JACOBI_ITER_;
    }
    PARCEL_INT get_blockjacobi(){
      return BLOCK_JACOBI_NUMBLOCK_;
    }

    precon_param *get_precon_param(){
      return (this);
    }

    bool get_scaling_matrix(){
      return SCALING_MATRIX_;
    }

    bool get_scaling_vector(){
      return SCALING_VECTOR_;
    }

    // PARCEL_INT get_@(){
    //   return @_ ;
    // }

    // void set_(PARCEL_INT @){
    //   @_ = @;
    // }

  };

  template<
    class Tmat,        // 係数行列精度
    class Tvec,        // 入出力ベクトル精度
    class Tprecon_mat, // 前処理行列精度
    class Tprecon_vec, // 前処理ベクトル精度
    class Tprecon_calc // 前処理の演算精度
    //    class Tprecon_mat_calc // 前処理行列作成時の演算精度
    >
  class precon_matrix{
    // 前処理行列作成時の演算精度を指定できるよう拡張
    // (現状はTmatの精度で計算)

    // jacobi
    precon_jacobi<
      Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tmat,Tprecon_calc> precon_jacobi_;

    // point jacobi
    precon_pointjacobi<
      Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tmat,Tprecon_calc> precon_pointjacobi_;

    //block jacobi
    precon_blockjacobi<
      Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tmat,Tprecon_calc> precon_blockjacobi_;

    precon_param precon_param_;

  public:

    // precon_matrixを前処理ごとに複数用意して必要なパラメータを切り替える
    // 共通処理は抜き出す
    // 前処理用のパラメータはclassにまとめる
    precon_matrix(
		  matrix<Tmat> &matrix,
		  precon_param &precon_param
		  // PARCEL_INT precon=PRECON_JACOBI,
		  // PARCEL_INT precon=PRECON_POINTJACOBI,
		  //		  PARCEL_INT precon=PRECON_BLOCKJACOBI,
		  // PARCEL_INT precon=PRECON_NONE,
		  //		  PARCEL_INT JACOBI_ITER=1
		  ){
      precon_param_ = precon_param;
      size_t matrix_type = matrix.matrix_type();      
      if(matrix_type == CRS_MATRIX){
	parcel::crsMat<Tmat> &crsMat = *matrix.crsMat_data();
	init_precon(crsMat);
	return ;
      }
      if(matrix_type == DIA_MATRIX){
	parcel::diaMat<Tmat> &diaMat = *matrix.diaMat_data();
	init_precon(diaMat);
	return ;
      }
      
    }
    ~precon_matrix(){}

    void init_precon(
		     parcel::crsMat<Tmat> &crsMat
		     ){

      PARCEL_INT precon = precon_param_.get_precon_algo();
      PARCEL_INT num_block = precon_param_.get_blockjacobi();
      PARCEL_INT ITER = precon_param_.get_jacobi();
      bool SCALING_MATRIX = precon_param_.get_scaling_matrix();
      bool SCALING_VECTOR = precon_param_.get_scaling_vector();

      switch (precon) {
      case PRECON_NONE:
	return ;
      case PRECON_POINTJACOBI:
	precon_pointjacobi_.init(crsMat);
	return ;
      case PRECON_BLOCKJACOBI:
	precon_blockjacobi_.init(crsMat,num_block,SCALING_MATRIX,SCALING_VECTOR);
	return ;
      case PRECON_JACOBI:
	precon_jacobi_.init(crsMat,ITER,SCALING_MATRIX,SCALING_VECTOR);
	return ;
      }
      return ;
    }

    void init_precon(
		     parcel::diaMat<Tmat> &diaMat
		     ){

      PARCEL_INT precon = precon_param_.get_precon_algo();
      PARCEL_INT num_block = precon_param_.get_blockjacobi();
      PARCEL_INT ITER = precon_param_.get_jacobi();
      bool SCALING_MATRIX = precon_param_.get_scaling_matrix();
      bool SCALING_VECTOR = precon_param_.get_scaling_vector();

      switch (precon) {
      case PRECON_NONE:
	return ;
      case PRECON_POINTJACOBI:
	precon_pointjacobi_.init(diaMat);
	return ;
      case PRECON_BLOCKJACOBI:
	precon_blockjacobi_.init(diaMat,num_block,SCALING_MATRIX,SCALING_VECTOR);
	return ;
      case PRECON_JACOBI:
	precon_jacobi_.init(diaMat,ITER,SCALING_MATRIX,SCALING_VECTOR);
	return ;
      }
      return ;
    }

    // solve Mz=r

    void solver(
		parcel::vector<Tvec> &r,parcel::vector<Tvec> &z){
      PARCEL_INT precon_algo = precon_param_.get_precon_algo();
      vector_util vec_util;

      switch (precon_algo) {
      case PRECON_NONE:
	vec_util.convert<Tvec,Tvec>(r,z);	
	return ;
      case PRECON_POINTJACOBI:
	precon_pointjacobi_.solver(r,z);
	return ;
      case PRECON_BLOCKJACOBI:
#if 0
	vec_util.convert<Tvec,Tvec>(r,z);	
#else
	precon_blockjacobi_.solver(r,z);
#endif
	return ;
      case PRECON_JACOBI:	
#if 0
	vec_util.convert<Tvec,Tvec>(r,z);	
#else
	precon_jacobi_.solver(r,z);
#endif
	return ;
      }
      return ;
    }
  };

}

#endif
