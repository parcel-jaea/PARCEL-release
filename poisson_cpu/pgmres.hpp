#ifndef PGMRES_HPP_
#define PGMRES_HPP_

// #include<quadmath.h>

#include <iomanip>
#include <iostream>
#include "mat.hpp"
#include "vec.hpp"
#include "krylov_precon.hpp"

template<class T>
class pgmres{

  std::vector<T> r_;
  std::vector< std::vector<T> > V_;
  std::vector<T> vi_;
  std::vector<T> pv_;
  std::vector<T> w_;

  vec<T> vec_;
  krylov_precon<T> precon_;
  PARCEL_INT itr_=0;
  // 相対残差ノルム
  T rnorm_ = 0.0; // ||r|| / ||b||

  PARCEL_INT itrmax_ =100;
  PARCEL_INT itrmax_jacobi_ = 4;
  T residualnorm_ = 1.0e-8;

  // GMRES(m)リスタート
  PARCEL_INT gmres_m_ = 4;
public:
  //  pgmres(crs_mat<T> &crsA){
  pgmres(){}
  ~pgmres(){}

  void set_residualnorm(T residualnorm){
    residualnorm_ = residualnorm;
  }
  void set_itrmax(PARCEL_INT itrmax){
    itrmax_ = itrmax;
  }

  void set_restart(PARCEL_INT restart){
    gmres_m_ = restart;
  }

  PARCEL_INT get_itrationNumber(){
    return itr_;
  }

  PARCEL_INT get_restart(){
    return gmres_m_;
  }
  
  T get_residualnorm(){
    return rnorm_;
  }

  void preconditioning(
		     std::vector<T> &x,
		     std::vector<T> &y
		     // std::vector<T> &x,
		     // std::vector<T> &y
		       ){
    // 前処理無し
    vec_.axpy(1.0,0.0,x,y);
    // ヤコビ前処理
    //precon_.jacobi_method(y,x,itrmax_jacobi_);
    // 対称ガウスザイデル
    //precon_.Symmetric_GaussSeidel_method(y,x,itrmax_jacobi_);
  }

  void solve(
	     crs_mat<T> &crsA,
	     std::vector<T> &x,
	     std::vector<T> &b
	     ){
    precon_.init(crsA);
    PARCEL_INT nRow = crsA.getValue_nRow();
    //    precon_.init(nRow);
    r_.resize(nRow,0.0);
    V_.resize(gmres_m_+1);
    for(PARCEL_INT i = 0;i < (gmres_m_+1);i++){
      V_[i].resize(nRow,0.0);
    }

    pv_.resize(nRow,0.0);
    w_.resize(nRow,0.0);
    
    // pgmres algorithm
    T bnorm = vec_.dot(b,b);
    bnorm = std::sqrt((double)bnorm);

    // ------
    // pgmres 
    // ------
    // do itr=1,....
    //   r=b-Ax
    //   v_1=r/||r||
    //   do i=1,m
    //     solve Mpv=v_i
    //     w=Apv
    //     グラムシュミット直交化
    //     v_i+1 = w/||w||
    //     ギブンス回転
    //     if(||r||<rmax)exit
    //   enddo
    //   解ベクトル更新
    //   if(||r||<rmax)exit
    // enddo

    for(PARCEL_INT itr = 0;itr < itrmax_/gmres_m_;itr++){
    //    for(PARCEL_INT itr = 0;itr < 10;itr++){
    //    for(PARCEL_INT itr = 0;itr < 1;itr++){
      // r = b - Ax
      crsA.SpMV(x,r_);
      vec_.axpy(1.0,-1.0,b,r_);

      rnorm_ = vec_.dot(r_,r_);
      rnorm_ = std::sqrt((double)rnorm_);

      //      rnorm_ = rnorm_ / bnorm;
      std::cout<<std::setw(8)<<1+itr*gmres_m_;
      std::cout<<" : residual norm = "<<(double)rnorm_/(double)bnorm<<std::endl;

      // V0 = r / || r ||
      T rnorm_inv = 1.0 / rnorm_;
      vec_.axpy(rnorm_inv,0.0,r_,V_[0]);

      std::vector<T> hki_((gmres_m_+1)*gmres_m_,0.0);
      std::vector<T> Rtri_((gmres_m_+1)*(gmres_m_+1),0.0);
      std::vector<T> e_(gmres_m_+1,0.0);
      std::vector<T> si_(gmres_m_+1,0.0);
      std::vector<T> co_(gmres_m_+1,0.0);

      e_[0] = rnorm_;
      for(PARCEL_INT m = 0;m < gmres_m_;m++){
	//     solve Mpv=v_i
	preconditioning(V_[m],pv_);
	//     w=Apv
	// crsA.SpMV(pv_,V_[m+1]);
	crsA.SpMV(pv_,w_);
	//     グラムシュミット直交化
	for(PARCEL_INT k=0;k<=m;k++){
	  T hkm = vec_.dot(w_,V_[k]);
	  hki_[ k + m*(gmres_m_+1) ] = hkm;
	  vec_.axpy(-hkm,1.0,V_[k],w_);
	}
	//     v_i+1 = w/||w||
	T wnorm = vec_.dot(w_,w_);
	wnorm = std::sqrt((double)wnorm);
	hki_[ m+1 + m*(gmres_m_+1) ] = wnorm;
	T wnorm_inv = 1.0/wnorm;
	vec_.axpy(wnorm_inv,0.0,w_,V_[m+1]);

#if 0
	// 直交化チェック	
	std::cout<<""<<std::endl;
	for(PARCEL_INT k=0;k<=m+1;k++){
	  for(PARCEL_INT i=k;i<=m+1;i++){
	    std::cout<<" <V,V> @ "<<i<<","<<k<<" : "<<vec_.dot(V_[k],V_[i])<<std::endl;
	  }
	}
	// ヘッセンベルグ行列
	for(PARCEL_INT k=0;k<gmres_m_;k++){
	  for(PARCEL_INT i=0;i<gmres_m_+1;i++){
	    std::cout<<" Heassenberg matrix : "<<k<<","<<i<<" : "<<hki_[i+k*(gmres_m_+1)]<<std::endl;
	  }
	  std::cout<<std::endl;
	}
#endif
	// ギブンス回転
	for(PARCEL_INT k=0;k<m;k++){
	  T tmp1 = hki_[k+m*(gmres_m_+1)]*co_[k] - hki_[(k+1)+m*(gmres_m_+1)]*si_[k];
	  T tmp2 = hki_[k+m*(gmres_m_+1)]*si_[k] + hki_[(k+1)+m*(gmres_m_+1)]*co_[k];
	  hki_[(k  )+m*(gmres_m_+1)] = tmp1;
	  hki_[(k+1)+m*(gmres_m_+1)] = tmp2;
	}
	T tmp3 = hki_[(m+1)+m*(gmres_m_+1)] / hki_[(m  )+m*(gmres_m_+1)];
	co_[m] = 1.0/sqrt(1.0+tmp3*tmp3 );
	si_[m] = -co_[m]*tmp3;
	hki_[(m  )+m*(gmres_m_+1)] = co_[m] * hki_[m+m*(gmres_m_+1)] - si_[m]* hki_[m+1+m*(gmres_m_+1)];
	hki_[(m+1)+m*(gmres_m_+1)] = 0.0;
	T tmp4 = co_[m]*e_[m] - si_[m]*e_[m+1];
	T tmp5 = si_[m]*e_[m] + co_[m]*e_[m+1];
	e_[m]   = tmp4;
	e_[m+1] = tmp5;
      }
      //   解ベクトル更新
#if 0
	      for(PARCEL_INT i=0;i<3 ;i++){
		std::cout<<"     check Q ";
		for(PARCEL_INT j=0;j<gmres_m_;j++){
		  std::cout<<V_[j][i]<<",";
		}
		std::cout<<std::endl;
	      }
#endif
      
      // solve y = hki e
#if 0
      // ヘッセンベルグ行列
	for(PARCEL_INT k=0;k<gmres_m_;k++){
	  for(PARCEL_INT i=0;i<gmres_m_+1;i++){
	    std::cout<<" Heassenberg matrix : "<<k<<","<<i<<" : "<<hki_[i+k*(gmres_m_+1)]<<std::endl;
	  }
	  std::cout<<std::endl;
	}
#endif	
#if 0
      for(PARCEL_INT k=0;k<gmres_m_+1;k++){
	std::cout<<"e_ @ "<<k<<","<<e_[k]<<","<<e_[k]/bnorm<<std::endl;
      }
      for(PARCEL_INT k=0;k<gmres_m_+1;k++){
	for(PARCEL_INT i=0;i<gmres_m_+1;i++){
	  std::cout<<k<<","<<i<<" : "<<hki_[i+k*(gmres_m_+1)]<<std::endl;
	}
	std::cout<<std::endl;
      }
#endif

#if 0
      for(PARCEL_INT k=0;k<gmres_m_;k++){
	std::cout<<"    zeta : "<<k<<" : "<<e_[k]<<std::endl;
      }
#endif      
      std::vector<T> y_(gmres_m_,0.0);
      for(PARCEL_INT k=gmres_m_-1;0<=k;k--){
	T y_sum = 0.0;
	for(PARCEL_INT i=k+1;i<gmres_m_;i++){
	  y_sum = y_sum + hki_[k+i*(gmres_m_+1)]*y_[i];
	}	
	y_[k] = (e_[k] - y_sum)/hki_[k+k*(gmres_m_+1)];
	// std::cout<<"     y: "<<k<<" : "<<y_[k]<<std::endl;
      }

      for(PARCEL_INT k=0;k<gmres_m_;k++){
	// std::cout<<k<<" : "<<y_[k]<<std::endl;
	if(k==0){
	  vec_.axpy(y_[k],0.0,V_[k],w_);
	}else{
	  vec_.axpy(y_[k],1.0,V_[k],w_);
	}
      }
      preconditioning(w_,pv_);
      vec_.axpy(1.0,1.0,pv_,x);
      //   if(||r||<rmax)exit      
    }

  }

};

#endif
