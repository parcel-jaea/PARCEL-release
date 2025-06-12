#ifndef MAT_HPP_
#define MAT_HPP_

// #ifndef PARCEL_INT
// #define PARCEL_INT int32_t
// #else
// #define PARCEL_INT int64_t
// #endif

#include <iostream>
#include <cstdint>
#include <vector>

template<class T>
class crs_mat{
  std::vector<T> A_;
  std::vector<T> Diag_;
  std::vector<PARCEL_INT> crsRow_ptr_;
  std::vector<PARCEL_INT> crsCol_;
  PARCEL_INT nRow_;
  PARCEL_INT nnz_;
public:
  void resize(PARCEL_INT n,PARCEL_INT nnz){
    nRow_ = n;
    nnz_  = nnz;
    A_.resize(nnz,0.0);
    Diag_.resize(n,0.0);
    crsRow_ptr_.resize(n+1,0);
    crsCol_.resize(nnz,0);
  }
  crs_mat(){}
  crs_mat(PARCEL_INT n,PARCEL_INT nnz){
    resize(n,nnz);
  };
  ~crs_mat(){}

  void reset(){
    A_.resize(1,0.0);
    Diag_.resize(1,0.0);
    crsRow_ptr_.resize(1,0);
    crsCol_.resize(1,0);
  }

  void setValue_nnz(PARCEL_INT nnz){
    nnz_ = nnz;
  }
  void setValue_nRow(PARCEL_INT n){
    nRow_ = n;
  }
  void setValue_matA(T val,PARCEL_INT id){
    A_[id] = val;
  }
  void setValue_matDiag(T val,PARCEL_INT id){
    Diag_[id] = val;
  }
  void setValue_crsRow_ptr(PARCEL_INT ival,PARCEL_INT id){
    crsRow_ptr_[id] = ival;
  }
  void setValue_crsCol(PARCEL_INT ival,PARCEL_INT id){
    crsCol_[id] = ival;
  }
  PARCEL_INT getValue_nRow(){
    return nRow_;
  }

  void SpMV(
	    std::vector<T> &x,
	    std::vector<T> &y
	    ){
#pragma omp parallel for
    for(auto i = 0; i < nRow_; i++){
      T tmp = 0.0;
      for(auto j = crsRow_ptr_[i]; j < crsRow_ptr_[i+1]; j++){
	auto col = crsCol_[j];
	tmp = tmp + A_[j]*x[col];
      }
      y[i] = tmp;
    }
  }

  
  void show(
	    PARCEL_INT nstart,
	    PARCEL_INT nend	    
	    ){
    for(auto i = nstart; i <= nend; i++){
      for(auto j = crsRow_ptr_[i]; j < crsRow_ptr_[i+1]; j++){
	auto col = crsCol_[j];
	std::cout<<" crs : "<<i<<", "<<col<<" : "<<A_[j]<<std::endl;
      }
    }
  }

  std::vector<T> A_ptr(){
    return A_;
  }
  std::vector<T> Diag_ptr(){
    return Diag_;
  }
  std::vector<PARCEL_INT> crsRow_ptr(){
    return crsRow_ptr_;
  }
  std::vector<PARCEL_INT> crsCol_ptr(){
    return crsCol_;
  }

  T *A_data(){
    return A_.data();
  }
  PARCEL_INT *crsRowptr_data(){
    return crsRow_ptr_.data();
  }
  PARCEL_INT *crsCol_data(){
    return crsCol_.data();
  }

};

template<class T>
class dia_mat{
  std::vector<T> A_;
  std::vector<T> Diag_;
  std::vector<PARCEL_INT> offset_;
  PARCEL_INT nRow_;
  PARCEL_INT nnd_;
  PARCEL_INT max_up_offset_;
  PARCEL_INT max_down_offset_;
public:
  void resize(PARCEL_INT n,PARCEL_INT nnd){
    nRow_ = n;
    nnd_  = nnd;
    A_.resize(n*nnd,0.0);
    Diag_.resize(n,0.0);
    offset_.resize(nnd,0);
    max_up_offset_   = 0;
    max_down_offset_ = 0;
  }
  dia_mat(){}
  dia_mat(PARCEL_INT n,PARCEL_INT nnd){
    resize(n,nnd);
  }
  ~dia_mat(){}

  void reset(){
    A_.resize(1,0.0);
    Diag_.resize(1,0.0);
    offset_.resize(1,0);
  }

  void setValue_nnd(PARCEL_INT nnd){
    nnd_ = nnd;
  }
  void setValue_nRow(PARCEL_INT n){
    nRow_ = n;
  }
  void setValue_matA(T val,PARCEL_INT id){
    A_[id] = val;
  }
  void setValue_matDiag(T val,PARCEL_INT id){
    Diag_[id] = val;
  }
  void setValue_offset(PARCEL_INT ival,PARCEL_INT id){
    offset_[id] = ival;
    max_up_offset_   = max_up_offset_   > ival ? ival : max_up_offset_;
    max_down_offset_ = max_down_offset_ < ival ? ival : max_down_offset_;
  }
  PARCEL_INT getValue_nRow(){
    return nRow_;
  }

  void SpMV(
	    std::vector<T> &x,
	    std::vector<T> &y
	    ){
    // std::cout<<"  nRow            : "<<nRow_<<std::endl;
    // std::cout<<"  max_up_offset   : "<<max_up_offset_<<std::endl;
    // std::cout<<"  max_down_offset : "<<max_down_offset_<<std::endl;

#if 1

    for(auto i = 0; i < nRow_; i++){
      y[i] = 0.0;
      if( ( (i+max_up_offset_)<0 ) || ( (i+max_down_offset_) >= nRow_ ) ){
	for(auto j = 0; j < nnd_; j++){
	  if( (0<=(i+offset_[j]) ) && ((i+offset_[j])<nRow_) ){
	    y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
	  }
	}
      }else{
	for(auto j = 0; j < nnd_; j++){
	  //	  if( (0<=(i+offset_[j]) ) && ((i+offset_[j])<nRow_) ){
	  y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
	  //	  }
	}
      }

    }

#endif

#if 0

    for(auto i = 0; i < 0-max_up_offset_; i++){
      y[i] = 0.0;
      for(auto j = 0; j < nnd_; j++){
	if( (0<=(i+offset_[j]) ) && ((i+offset_[j])<nRow_) ){
	  y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
	}
      }
    }
    for(auto i = 0-max_up_offset_; i < nRow_-max_down_offset_; i++){
      y[i] = 0.0;
      for(auto j = 0; j < nnd_; j++){
	y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
      }
    }
    for(auto i = nRow_-max_down_offset_; i < nRow_; i++){
      y[i] = 0.0;
      for(auto j = 0; j < nnd_; j++){
	if( (0<=(i+offset_[j]) ) && ((i+offset_[j])<nRow_) ){
	  y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
	}
      }
    }

#endif

#if 0
    for(auto i = 0; i < nRow_; i++){
      y[i] = 0.0;
      for(auto j = 0; j < nnd_; j++){
	if( (0<=(i+offset_[j]) ) && ((i+offset_[j])<nRow_) ){
	  y[i] = y[i] + A_[i+j*nRow_]*x[i+offset_[j]];
	}
      }
    }
#endif

  }

  
  std::vector<T> A_ptr(){
    return A_;
  }
  std::vector<T> Diag_ptr(){
    return Diag_;
  }
  std::vector<PARCEL_INT> offset_ptr(){
    return offset_;
  }

  T *A_data(){
    return A_.data();
  }
  PARCEL_INT *offset_data(){
    return offset_.data();
  }

};

template<class T>
class sparce_matrix{
  const PARCEL_INT CRS_MATRIX = 1;
  const PARCEL_INT DIA_MATRIX = 2;
  PARCEL_INT matrix_type;
  crs_mat<T> crsMat_;
  dia_mat<T> diaMat_;
 public:
   sparce_matrix(){
     matrix_type = -1;
   }
  ~sparce_matrix(){}

  void set_crsMat(PARCEL_INT n,PARCEL_INT nnz){
    matrix_type = CRS_MATRIX;
    crsMat_.resize(n,nnz);
  }

  void set_diaMat(PARCEL_INT n,PARCEL_INT nnz){
    matrix_type = DIA_MATRIX;
    diaMat_.resize(n,nnz);
  }

  crs_mat<T>  *crsMat_data(){
    return &crsMat_;
  }
  dia_mat<T>  *diaMat_data(){
    return &diaMat_;
  }

};

#endif
