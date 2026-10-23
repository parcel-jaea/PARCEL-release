// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_PRECON_HPP_
#define PARCEL_PRECON_HPP_

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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
    // 利用者が set_blockjacobi を呼んだかどうか。呼ばれていなければ
    // デバイスの並列度から自動で決める (precon_matrix::blockjacobi_default)。
    bool BLOCK_JACOBI_NUMBLOCK_SET_ = false;
    bool SCALING_MATRIX_ = false;
    bool SCALING_VECTOR_ = false;
    // 利用者が set_scaling_* を呼んだかどうか。呼ばれていなければ
    // 前処理の精度と種類から自動で決める (precon_matrix::scaling_default)。
    bool SCALING_MATRIX_SET_ = false;
    bool SCALING_VECTOR_SET_ = false;

  public :
    precon_param(){}
    ~precon_param(){}
    // 前処理の種類として受け付ける値か (0, 1, 2, 4)
    static bool valid_precon_algo(PARCEL_INT a){
      return a == PRECON_NONE || a == PRECON_POINTJACOBI ||
	     a == PRECON_BLOCKJACOBI || a == PRECON_JACOBI;
    }
    // 設定値を検査して std::invalid_argument を投げる (範囲外の値で黙って前処理なしにしない)。
    //   設定関数は全ランクで同じ引数で呼ばれる前提なので、全ランクが同じように投げる。
    void set_precon_algo(PARCEL_INT precon_algo){
      if(!valid_precon_algo(precon_algo)){
	throw std::invalid_argument("parcel::precon_param::set_precon_algo: invalid value "
				    + std::to_string((long long)precon_algo)
				    + " (expected 0=NONE, 1=POINTJACOBI, 2=BLOCKJACOBI, 4=JACOBI)");
      }
      precon_algo_ = precon_algo;
    }
    void set_jacobi(PARCEL_INT ITER){
      if(ITER < 1){
	throw std::invalid_argument("parcel::precon_param::set_jacobi: iteration count must be >= 1 (got "
				    + std::to_string((long long)ITER) + ")");
      }
      JACOBI_ITER_ = ITER;
    }
    void set_blockjacobi(PARCEL_INT BLOCK_JACOBI_NUMBLOCK){
      if(BLOCK_JACOBI_NUMBLOCK < 1){
	throw std::invalid_argument("parcel::precon_param::set_blockjacobi: number of blocks must be >= 1 (got "
				    + std::to_string((long long)BLOCK_JACOBI_NUMBLOCK) + ")");
      }
      BLOCK_JACOBI_NUMBLOCK_     = BLOCK_JACOBI_NUMBLOCK;
      BLOCK_JACOBI_NUMBLOCK_SET_ = true;
    }
    bool blockjacobi_is_set(){ return BLOCK_JACOBI_NUMBLOCK_SET_; }

    void set_scaling_matrix( bool SCALING_MATRIX){
      SCALING_MATRIX_     = SCALING_MATRIX;
      SCALING_MATRIX_SET_ = true;
      return ;
    }

    void set_scaling_vector( bool SCALING_VECTOR){
      SCALING_VECTOR_     = SCALING_VECTOR;
      SCALING_VECTOR_SET_ = true;
      return ;
    }

    bool scaling_matrix_is_set(){ return SCALING_MATRIX_SET_; }
    bool scaling_vector_is_set(){ return SCALING_VECTOR_SET_; }

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

  };

  template<
    class Tmat,        // 係数行列精度
    class Tvec,        // 入出力ベクトル精度
    class Tprecon_mat, // 前処理行列精度
    class Tprecon_vec, // 前処理ベクトル精度
    class Tprecon_calc // 前処理の演算精度
    >
  class precon_matrix{
    // 前処理行列の作成は Tmat の精度で計算する

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

    // 前処理の精度からスケーリングの要否を決める。
    //
    //   スケーリングは max() で求めた値で行列・残差を割る処理で、
    //   半精度の表現範囲 (最大 65504) に収めるためのもの。
    //   倍精度・単精度では不要で、max() の呼び出しと除算のぶん遅くなるだけ
    //   (外すと GPU で前処理時間が約 3 割減る。反復回数はほぼ変わらない)。
    static constexpr bool precision_needs_scaling(){
      using Tcalc_base = typename std::result_of<
	parcel::parcel_type_to_baseType(Tprecon_calc)>::type;
      using Tmat_base  = typename std::result_of<
	parcel::parcel_type_to_baseType(Tprecon_mat)>::type;
      return (sizeof(Tcalc_base) <= 2) || (sizeof(Tmat_base) <= 2);
    }

    // ブロック Jacobi の分割数。set_blockjacobi() が呼ばれていないときの既定。
    //
    //   適用カーネルの並列度は分割数にそのまま等しい
    //   (parcel_precon_blockjacobi.hpp の parallel_for(range<1>{分割数}))。
    //   分割数 1 は work item 1 個を意味し、デバイスをまったく使えない。
    //
    //   CPU: ランクあたりのスレッド数にする。ブロックを細かくするほど前処理は
    //        弱くなるので、並列度が埋まる最小の分割数が最適になる。
    //
    //   GPU: 数万スレッドを埋めるには 1 ブロック数行まで分割する必要があり、
    //        そこまで細かくすると前処理としてほとんど効かない (Jacobi の方が速い)。
    //        BJ を選んだ場合に破綻しない値として 1 ブロック 2 行を既定にする。
    static PARCEL_INT blockjacobi_default(sycl::queue *q, size_t n){
      if(q != nullptr && q->get_device().is_gpu()){
	const PARCEL_INT b = (PARCEL_INT)(n / 2);
	return b > 0 ? b : 1;
      }
      const PARCEL_INT b = (PARCEL_INT)parcel::nblock_runtime();
      return b > 0 ? b : 1;
    }

    // set_scaling_* が呼ばれていないときの既定値。
    //   ブロック Jacobi は精度によらず有効にする。対流優勢な問題では
    //   スケーリングを外すと収束しなくなる。
    static bool scaling_default(PARCEL_INT precon){
      if(precon == PRECON_BLOCKJACOBI){ return true; }
      return precision_needs_scaling();
    }

  public:

    precon_matrix(
		  matrix<Tmat> &matrix,
		  precon_param &precon_param
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
      // 明示指定がなければデバイスの並列度から決める
      PARCEL_INT num_block = precon_param_.blockjacobi_is_set()
	? precon_param_.get_blockjacobi()
	: blockjacobi_default(crsMat.queue(), crsMat.n());
      PARCEL_INT ITER = precon_param_.get_jacobi();
      // 明示指定がなければ精度と前処理の種類で決める
      bool SCALING_MATRIX = precon_param_.scaling_matrix_is_set()
	? precon_param_.get_scaling_matrix() : scaling_default(precon);
      bool SCALING_VECTOR = precon_param_.scaling_vector_is_set()
	? precon_param_.get_scaling_vector() : scaling_default(precon);
      // 自動判定の結果は外から見えないので、確認したいときだけ出せるようにする
      if(std::getenv("PARCEL_PRECON_VERBOSE")){
	std::cout<<" [precon] algo = "<<(int)precon
		 <<" , scaling_matrix = "<<SCALING_MATRIX
		 <<" , scaling_vector = "<<SCALING_VECTOR
		 <<" (precision_needs_scaling = "<<precision_needs_scaling()<<")"
		 <<" , bj_block = "<<(long)num_block
		 <<(precon_param_.blockjacobi_is_set() ? " (指定)" : " (自動)")
		 <<std::endl;
      }

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
      default:
	// 未知の値で黙って前処理なしにしない
	throw std::invalid_argument("parcel::precon_matrix::init_precon: unknown precon_algo "
				    + std::to_string((long long)precon));
      }
      return ;
    }

    void init_precon(
		     parcel::diaMat<Tmat> &diaMat
		     ){

      PARCEL_INT precon = precon_param_.get_precon_algo();
      // 明示指定がなければデバイスの並列度から決める
      PARCEL_INT num_block = precon_param_.blockjacobi_is_set()
	? precon_param_.get_blockjacobi()
	: blockjacobi_default(diaMat.queue(), diaMat.n());
      PARCEL_INT ITER = precon_param_.get_jacobi();
      // 明示指定がなければ精度と前処理の種類で決める
      bool SCALING_MATRIX = precon_param_.scaling_matrix_is_set()
	? precon_param_.get_scaling_matrix() : scaling_default(precon);
      bool SCALING_VECTOR = precon_param_.scaling_vector_is_set()
	? precon_param_.get_scaling_vector() : scaling_default(precon);
      // 自動判定の結果は外から見えないので、確認したいときだけ出せるようにする
      if(std::getenv("PARCEL_PRECON_VERBOSE")){
	std::cout<<" [precon] algo = "<<(int)precon
		 <<" , scaling_matrix = "<<SCALING_MATRIX
		 <<" , scaling_vector = "<<SCALING_VECTOR
		 <<" (precision_needs_scaling = "<<precision_needs_scaling()<<")"
		 <<" , bj_block = "<<(long)num_block
		 <<(precon_param_.blockjacobi_is_set() ? " (指定)" : " (自動)")
		 <<std::endl;
      }

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
      default:
	// 未知の値で黙って前処理なしにしない
	throw std::invalid_argument("parcel::precon_matrix::init_precon: unknown precon_algo "
				    + std::to_string((long long)precon));
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
	precon_blockjacobi_.solver(r,z);
	return ;
      case PRECON_JACOBI:	
	precon_jacobi_.solver(r,z);
	return ;
      default:
	// 未知の値で黙って z を未設定のまま返さない
	throw std::invalid_argument("parcel::precon_matrix::solver: unknown precon_algo "
				    + std::to_string((long long)precon_algo));
      }
      return ;
    }
  };

}

#endif
