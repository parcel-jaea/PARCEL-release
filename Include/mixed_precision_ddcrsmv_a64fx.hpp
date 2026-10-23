// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDCRSMV_A64FX_HPP_
#define MIXED_PRECISON_DDCRSMV_A64FX_HPP_

void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi    ,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi ,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  ddmul_calc1(A_hi[j],A_lo[j],x_hi[col],x_lo[col],work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
      }

    });
        
  return ;

}

template< class TdataA ,class TdataB >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha_hi,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy, 
 	     double *dwork_,size_t work_bloncksize){      


  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = 0.0;
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = 0.0;
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  ddmul_calc1(A_hi[j],A_lo[j],x_hi[col],x_lo[col],work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
      }

    });
        
  return ;
  
}

template< class TdataCRS >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     TdataCRS *A_hi,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      


  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double A_lo_ = 0.0;
	  ddmul_calc1(A_hi[j],A_lo_,x_hi[col],x_lo[col],work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
      }

    });
        
  return ;

}

template< class TdataA ,class TdataB,  class TdataCRS  >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha_hi,
	     TdataCRS *A_hi,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy ,
	     double *dwork_,size_t work_bloncksize){      


  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = 0.0;
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = 0.0;
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double A_lo_=0.0;
	  ddmul_calc1(A_hi[j],A_lo_,x_hi[col],x_lo[col],work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	}
      }

    });
        
  return ;

}

template< class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y_hi,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      


  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double x_lo_ = 0.0;
	  ddmul_calc1(A_hi[j],A_lo[j],x_hi[col],x_lo_,work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	double by_hi[PARCEL_SPMV_TILE];
	double by_lo[PARCEL_SPMV_TILE];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double y_lo_ = 0.0;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo_,by_hi[i-iii],by_lo[i-iii]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double y_lo_ = 0.0;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	}
      }

    });
        
  return ;

}

template<class TdataA ,class TdataB,class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha_hi,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     TdataY *y_hi,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = 0.0;
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = 0.0;
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double A_lo_ = 0.0;
	  double x_lo_ = 0.0;
	  ddmul_calc1(A_hi[j],A_lo_,x_hi[col],x_lo_,work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	double by_hi[PARCEL_SPMV_TILE];
	double by_lo[PARCEL_SPMV_TILE];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double y_lo_ = 0.0;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo_,by_hi[i-iii],by_lo[i-iii]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double y_lo_ = 0.0;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	}
      }

    });
        
  return ;

}

template< class TdataCRS, class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     TdataCRS *A_hi,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y_hi,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double A_lo_=0.0;
	  double x_lo_=0.0;
	  ddmul_calc1(A_hi[j],A_lo_,x_hi[col],x_lo_,work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}
	       
	double by_hi[PARCEL_SPMV_TILE];
	double by_lo[PARCEL_SPMV_TILE];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double y_lo_ = 0.0;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo_,by_hi[i-iii],by_lo[i-iii]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double y_lo_ = 0.0;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	}
      }

    });
        
  return ;

}

template< class TdataA,class TdataB, class TdataCRS, class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha_hi,
	     TdataCRS *A_hi,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     TdataY *y_hi,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  const size_t MAX   = N;

  // スレッド分割は連続、work[] の粒度はタイル。mixed_precision_crsmv_a64fx.hpp と同じ形
  //   (理由は parcel_crs.hpp の PARCEL_SPMV_TILE のコメントを参照)。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;
      
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = 0.0;
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = 0.0;
      const size_t bb = ii*per;
      const size_t ee = (bb+per) > (size_t)N ? (size_t)N : (bb+per);
      for(size_t iii=bb; iii<ee; iii+=TILE){
	const size_t imax = (iii+TILE) > ee ? ee : (iii+TILE);
	double aAx_hi[PARCEL_SPMV_TILE];
	double aAx_lo[PARCEL_SPMV_TILE];
	double *work_hi = &dwork_[(ii*2)*work_bloncksize];
	double *work_lo = &dwork_[(ii*2)*work_bloncksize + work_bloncksize];
	double ddadd_var1;
	for(PARCEL_LINT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  double ddmul_var1;
	  const PARCEL_LINT col = crsCol[j];
	  double A_lo_ = 0.0;
	  double x_lo_ = 0.0;
	  ddmul_calc1(A_hi[j],A_lo_,x_hi[col],x_lo_,work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double Ax1_hi = 0.0;
	  double Ax1_lo = 0.0;
	  double Ax2_hi = 0.0;
	  double Ax2_lo = 0.0;
	  PARCEL_LINT j0 = 0;
	  PARCEL_LINT Colblock = Rowptr[i+1]-Rowptr[i];
	  if( ( Colblock%2 ) == 1){
	    j0 =1;
	    PARCEL_LINT j = Rowptr[i];
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]],work_lo[j-Rowptr[iii]] ,Ax1_hi,Ax1_lo);
	    Colblock = Colblock-1;		   
	  }
	  Colblock = Colblock/2;
	  for(PARCEL_LINT j = Rowptr[i]+j0; j < Rowptr[i]+j0+Colblock; j++){
	    double ddadd_var1;
	    ddadd_calc1(Ax1_hi,Ax1_lo, work_hi[j-Rowptr[iii]]           ,work_lo[j-Rowptr[iii]]          ,Ax1_hi,Ax1_lo);
	    double ddadd_var2;
	    ddadd_calc2(Ax2_hi,Ax2_lo, work_hi[j+Colblock-Rowptr[iii]]  ,work_lo[j+Colblock-Rowptr[iii]] ,Ax2_hi,Ax2_lo);
	  }
	  double Ax_hi = 0.0;
	  double Ax_lo = 0.0;
	  ddadd_calc1(Ax1_hi,Ax1_lo, Ax2_hi,Ax2_lo, Ax_hi,Ax_lo );
	  aAx_hi[i-iii] = Ax_hi;
	  aAx_lo[i-iii] = Ax_lo;
	}

	       
	double by_hi[PARCEL_SPMV_TILE];
	double by_lo[PARCEL_SPMV_TILE];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double y_lo_ = 0.0;
	  ddmul_calc1(alpha_hi_,alpha_lo_,aAx_hi[i-iii],aAx_lo[i-iii],aAx_hi[i-iii],aAx_lo[i-iii]);
	  ddmul_calc1(beta_hi_,beta_lo_,y_hi[i],y_lo_,by_hi[i-iii],by_lo[i-iii]);
	}
	       
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double y_lo_ = 0.0;
	  ddadd_calc1(aAx_hi[i-iii] ,aAx_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	}

      }

    });
        
  return ;

}

#endif

