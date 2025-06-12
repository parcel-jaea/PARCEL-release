#ifndef PARCEL_HPP_
#define PARCEL_HPP_

// #define PARCEL_INT int64_t
// #define PARCEL_INT int32_t

// #define PARCEL_MAX_FLOTING_TYPE long double
#define PARCEL_MAX_FLOTING_TYPE double
#if PARCEL_MPI
#define PARCEL_MPI_MAXFLOTING_TYPE MPI_DOUBLE
#endif

// #define PARCEL_INT int

#include <type_traits>
#include <vector>
// #include "util_sycl.hpp"
// #include "ddlib.hpp"
#include "parcel_sycl.hpp"
#include "parcel_scalar.hpp"

namespace parcel{

  using ddtype = ddlib::ddtype;
  const size_t CRS_MATRIX = 1;
  const size_t DIA_MATRIX = 2;

  // -------------------------------
  template<class T>
  class floatingtype{
    util_sycl::vector<T> vec;
  public:
    
    floatingtype(){}
    ~floatingtype(){}

    void resize(size_t nnz, sycl::queue *q){
      vec.resize(nnz,q);
      return ;
    }

    sycl::queue *queue(){
      return vec.queue();
    }

    void setValue( std::vector< parcel::scalar > &mem_){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_[i].get_T();
      }
      vec.setValue(vec_x);
    }


    void setValue(parcel::scalar mem_){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_.get_T();
      }
      vec.setValue(vec_x);
    }

    template<class Tdata>
    void setValue(Tdata *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem[i];
      }
      vec.setValue(vec_x);
    }
    void setValue(parcel::scalar *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem[i].get_T();
      }
      vec.setValue(vec_x);
    }

    template<class Tdata>
    void setValue(Tdata *mem_hi,Tdata *mem_lo){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_hi[i] + mem_lo[i];
      }
      vec.setValue(vec_x);
    }

    void getValue( std::vector< parcel::scalar > &mem ){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      vec.getValue(vec_x);
      for(size_t i = 0;i<n;i++){
	PARCEL_MAX_FLOTING_TYPE vec_x_ = vec_x[i];
	mem[i].set_Tdd(vec_x_);
      }
    }
    
    template<class Tdata>
    void getValue(Tdata *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      vec.getValue(vec_x);
      for(size_t i = 0;i<n;i++){
	mem[i] = vec_x[i];
      }
    }

    void setValue(ddlib::ddtype mem_){
      T  mem = mem_.get_T<T>();
      vec.setValue(&mem);
    }

    void setValue(T *mem){
      vec.setValue(mem);
    }

    void getValue(T *mem){
      vec.getValue(mem);
    }

    void getValue_hi(T *mem){
      vec.getValue(mem);
    }
    void getValue_lo(T *mem){
      //vec.getValue(mem);
      mem = nullptr;
    }
    
    void getValue_d_hi(double *mem){
      std::vector<T> mem_T(vec.size());
      getValue(mem_T.data());
      for(PARCEL_INT i=0;i<vec.size();i++){
	  mem[i] = mem_T[i];
      }
      // vec.getValue(mem);
    }
    
    void getValue_d_lo(double *mem){
      //      vec.getValue(mem);
      mem = nullptr;
    }

    T *data(){
      return vec.data();
    }

    T *data_hi(){
      return vec.data();
    }

    T *data_lo(){
      return nullptr;
    }

    size_t size(){
      return  vec.size();
    }

    void shallowcp(T* mem, size_t n, sycl::queue* q){
      vec.shallowcp(mem,n,q);
      return ;
    }

    void shallowcp_hi(T* mem, size_t n, sycl::queue* q){
      vec.shallowcp(mem,n,q);
      return ;
    }

    void shallowcp_lo(T* mem, size_t n, sycl::queue* q){
      mem = nullptr;      
      return ;
    }

    // auto getType(){
    //   return ;
    // }

  };

  class doubledouble{
    util_sycl::vector<double> vec_hi;
    util_sycl::vector<double> vec_lo;
  public:

    doubledouble(){}
    ~doubledouble(){}

    void resize(size_t nnz, sycl::queue *q){
      vec_hi.resize(nnz,q);
      vec_lo.resize(nnz,q);
      return ;
    }

    sycl::queue *queue(){
      return vec_hi.queue();
    }

    size_t size(){
      return  vec_hi.size();
    }

    template<class T>
    void setValue_hi(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_(n,0.0);
      double *vec = vec_.data();
      for(size_t i = 0;i<n;i++){
	vec[i] = mem[i];
      }
      vec_hi.setValue(vec);
    }

    template<class T>
    void setValue_lo(T *mem){
      size_t n = vec_lo.size();
      std::vector<double> vec_(n,0.0);
      double *vec = vec_.data();
      for(size_t i = 0;i<n;i++){
	vec[i] = mem[i];
      }
      vec_lo.setValue(vec);
    }

    void setValue(double *mem_hi,double *mem_lo){
      vec_hi.setValue(mem_hi);
      vec_lo.setValue(mem_lo);
    }

    void setValue(parcel::scalar mem_){
      double mem_hi = mem_.get_dd_hi();
      double mem_lo = mem_.get_dd_lo();

      size_t n = vec_lo.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();

      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem_hi;
      }
      vec_hi.setValue(vec_data_hi);

      for(size_t i = 0;i<n;i++){
	vec_data_lo[i] = mem_lo;
      }
      vec_lo.setValue(vec_data_lo);

    }

    void setValue( std::vector< parcel::scalar > &mem_){
      // double mem_hi = mem_.get_dd_hi();
      // double mem_lo = mem_.get_dd_lo();

      size_t n = vec_lo.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();

      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem_[i].get_dd_hi();
      }
      vec_hi.setValue(vec_data_hi);

      for(size_t i = 0;i<n;i++){
	vec_data_lo[i] = mem_[i].get_dd_lo();
      }
      vec_lo.setValue(vec_data_lo);

    }

    template<class T>
    void setValue(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();
      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem[i];
	vec_data_lo[i] = 0.0;
      }
      vec_hi.setValue(vec_data_hi);
      vec_lo.setValue(vec_data_lo);
    }
    void setValue(parcel::scalar *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();
      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem[i].get_dd_hi();
	vec_data_lo[i] = mem[i].get_dd_lo();
      }
      vec_hi.setValue(vec_data_hi);
      vec_lo.setValue(vec_data_lo);
    }

    void setValue_hi(double *mem){
      vec_hi.setValue(mem);
    }

    void setValue_lo(double *mem){
      vec_lo.setValue(mem);
    }

    template<class T>
    void getValue(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> mem_hi_(n,0.0);
      std::vector<double> mem_lo_(n,0.0);
      double *mem_hi = mem_hi_.data();
      double *mem_lo = mem_lo_.data();

      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);

      for(size_t i = 0;i<n;i++){
	mem[i] = mem_hi[i] + mem_lo[i];
      }
    }

    void getValue(  std::vector< parcel::scalar > &mem ){
      size_t n = vec_hi.size();
      std::vector<double> mem_hi_(n,0.0);
      std::vector<double> mem_lo_(n,0.0);
      double *mem_hi = mem_hi_.data();
      double *mem_lo = mem_lo_.data();

      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);

      for(size_t i = 0;i<n;i++){
	mem[i].set_T(  mem_hi[i] + mem_lo[i] );
	mem[i].set_dd( mem_hi[i] , mem_lo[i] );
      }
    }

    void getValue_hi(double *mem){
      vec_hi.getValue(mem);
    }
    void getValue_lo(double *mem){
      vec_lo.getValue(mem);
    }
    void getValue(double *mem_hi,double *mem_lo){
      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);
    }
    void getValue( std::vector<double> &mem_hi, std::vector<double> &mem_lo){
      vec_hi.getValue(mem_hi.data());
      vec_lo.getValue(mem_lo.data());
    }

    double *data(){
      return vec_hi.data();
    }
    double *data_hi(){
      return vec_hi.data();
    }
    double *data_lo(){
      return vec_lo.data();
    }

    void shallowcp_hi(double* mem_hi, size_t n, sycl::queue* q){
      vec_hi.shallowcp(mem_hi,n,q);
      return ;
    }
    void shallowcp_lo(double* mem_lo, size_t n, sycl::queue* q){
      vec_lo.shallowcp(mem_lo,n,q);
      return ;
    }
    void shallowcp(double* mem_hi ,double* mem_lo, size_t n, sycl::queue* q){
      vec_hi.shallowcp(mem_hi,n,q);
      vec_lo.shallowcp(mem_lo,n,q);
	return ;
    }

    void shallowcp(double* mem_hi ,size_t n, sycl::queue* q){
      shallowcp_hi(mem_hi,n,q);
      return ;
    }

  };


template<class parceltype>
  class vector{
    parceltype mem_;
    size_t n_;
    //    double a_;
  public:
    // auto ret_a(){
    //   return a_;
    // }
    void resize(size_t n, sycl::queue *q){
      n_ = n;
      mem_.resize(n,q);
    }

    vector(){}
    //    vector(parceltype *mem,size_t n, sycl::queue* q){

    vector(size_t n, sycl::queue* q){
      resize(n,q);
    }
    ~vector(){}

    sycl::queue *queue(){
      return mem_.queue();
    }

    template<class T>
    void setValue(T *mem){
      mem_.setValue(mem);
    }
  
    template<class T>
    void setValue(std::vector<T> &mem){
      mem_.setValue(mem.data());
    }

    void setValue(parcel::scalar mem){
      mem_.setValue(mem);
    }

    void setValue( std::vector< parcel::scalar >  &mem){
      mem_.setValue(mem);
    }

    void setValue(double *mem_hi,double *mem_lo){
      mem_.setValue(mem_hi,mem_lo);
    }

    void setValue(std::vector<double> &mem_hi,std::vector<double> &mem_lo){
      mem_.setValue(mem_hi.data(),mem_lo.data());
    }

    void setValue_hi(double *mem_hi){
      mem_.setValue_hi(mem_hi);
    }
    void setValue_lo(double *mem_lo){
      mem_.setValue_lo(mem_lo);
    }

    template<class T>
    void getValue(T *mem){
      mem_.getValue(mem);
    }

    void getValue( std::vector< parcel::scalar >  &mem){
      mem_.getValue(mem);
    }

    template<class T>
    void getValue_hi(T *mem_hi){
      mem_.getValue_hi(mem_hi);
    }
    template<class T>
    void getValue_lo(T *mem_lo){
      mem_.getValue_lo(mem_lo);
    }
    
  template<class T>
  void getValue_hi( std::vector<T> &mem_hi){
    mem_.getValue_hi(mem_hi.data());
  }
  template<class T>
  void getValue_lo( std::vector<T> &mem_lo){
    mem_.getValue_lo(mem_lo.data());
  }

    void getValue_hi(double *mem_hi){
      mem_.getValue_hi(mem_hi);
    }
    void getValue_lo(double *mem_lo){
      mem_.getValue_lo(mem_lo);
    }

    // template<class T>
    // void getValue(double *mem_hi,double *mem_lo){
    //   mem_.getValue(mem);
    // }

    template<class T>
    void getValue(T *mem_hi,T *mem_lo){
      mem_.getValue_hi(mem_hi);
      mem_.getValue_lo(mem_lo);
    }
  
    void getValue(double *mem_hi,double *mem_lo){
      mem_.getValue_d_hi(mem_hi);
      mem_.getValue_d_lo(mem_lo);
    }
  
    size_t size(){
      return mem_.size();
    }

    parceltype *mem(){
      return &mem_;
    }

    auto *data(){
      return mem_.data();
    }
    auto *data_hi(){
      return mem_.data_hi();
    }
    auto *data_lo(){
      return mem_.data_lo();
    }

  };
}


#include "parcel_type.hpp"
#include "parcel_util.hpp"

namespace parcel{
  
template<class parceltype>
class parallel_vector{
  
  parcel::vector_util vec_util_;
  //  parcel::vector<parceltype> vec_;
  parcel::vector<parceltype> vec_add_;
  parcel::vector<parceltype> vec_;
  PARCEL_INT n_;
  PARCEL_INT nadd_;
  PARCEL_INT num_;
  PARCEL_INT ld_;
  
public :

  void resize(
	      PARCEL_INT n, PARCEL_INT nadd, PARCEL_INT num, sycl::queue *q
	      ){
    n_    = n;
    nadd_ = nadd;
    num_  = num;
    ld_   = n + nadd;    
    vec_add_.resize((n_+nadd_)*num_,q);
  }
  
  parallel_vector(){
    sycl::queue q_default{sycl::default_selector{}};
    resize(1,1,1,&q_default);
  }
  
  parallel_vector(
		  PARCEL_INT n, PARCEL_INT nadd, PARCEL_INT num, sycl::queue *q
		  ){
    resize(n,nadd,num,q);
  }
  
  ~parallel_vector(){}
  
  parcel::vector<parceltype> &vec(PARCEL_INT id){
    // parcel::vector<parceltype> vec_;
    vec_util_.setptr(n_,vec_add_,vec_,ld_*id);
    return vec_;
  }

  parcel::vector<parceltype> &vec_add(){
    // // parcel::vector<parceltype> vec_;
    // vec_util_.setptr(ld_*num_,vec_add_,vec_,0);    
    return vec_add_;
  }
  
  parcel::vector<parceltype> &vec_add(PARCEL_INT id){
    // parcel::vector<parceltype> vec_;
    vec_util_.setptr(ld_,vec_add_,vec_,ld_*id);
    return vec_;
  }

  parcel::vector<parceltype> &vec_add(PARCEL_INT id, PARCEL_INT num){
    // parcel::vector<parceltype> vec_;
    vec_util_.setptr(ld_*num,vec_add_,vec_,ld_*id);
    return vec_;
  }
  
};
  
}

#if PARCEL_MPI
#include "./parcel_mpi/parcel_mpi.hpp"
#endif

namespace parcel{
  template<class parceltype>
  class crsMat{
    // parceltype mat_;
    vector<parceltype> mat_;
    util_sycl::vector<PARCEL_INT> rowptr_;
    util_sycl::vector<PARCEL_INT> crsCol_;
    size_t n_; // ローカル行数
    size_t nnz_; // 非ゼロ要素数
    size_t maxCol_; // 最大列数
    
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};

#if PARCEL_MPI

    // MPI並列分散行列
    size_t gn_; // グローバル行数
    PARCEL_INT nstart_; // 開始行数
    PARCEL_INT nend_; // 開始行数
    
    parcel::comm *comm_;

    // 各プロセスの担当範囲
    std::vector<PARCEL_INT> nstart_list_;
    std::vector<PARCEL_INT> nend_list_;
    
    // 各プロセスへの送信データ数
    std::vector<PARCEL_INT> nsend_list_;
    // 各プロセスへの送信列番号リスト(グローバル列番号)
    std::vector< std::vector<PARCEL_INT> > nsend_gcol_list_;
    
    // 各プロセスからの受信データ数
    std::vector<PARCEL_INT> nrecv_list_;
    // 各プロセスからの受信列番号リスト(グローバル列番号)
    std::vector< std::vector<PARCEL_INT> > nrecv_gcol_list_;

    // グローバル列番号
    std::vector<PARCEL_INT> gcol_list_;

    // ------------------------------------------------------------
    // SYCL用変数
    std::vector< util_sycl::vector<PARCEL_MAX_FLOTING_TYPE> > send_comm_buff_;
    std::vector< util_sycl::vector<PARCEL_MAX_FLOTING_TYPE> > recv_comm_buff_;

    std::vector< std::vector<PARCEL_MAX_FLOTING_TYPE> > send_comm_hostbuff_;
    std::vector< std::vector<PARCEL_MAX_FLOTING_TYPE> > recv_comm_hostbuff_;
    
    // 各プロセスへの送信列番号リスト(グローバル列番号)
    std::vector< util_sycl::vector<PARCEL_INT> > nsend_gcol_parcel_list_;
    
    // 各プロセスからの受信列番号リスト(グローバル列番号)
    std::vector< util_sycl::vector<PARCEL_INT> > nrecv_gcol_parcel_list_;
    
#endif
    
  public:

#include "mixed_precision_comm_crs.hpp"
    
    void set_queue(sycl::queue* q){
      q_ = q;
      return ;    
    }

#if PARCEL_MPI

    template<class Tvecx>
    void SpMVcomm(
		  sycl::queue *que_,Tvecx *x
		  ){

      MPI_Comm mpi_comm = comm_->mpi_comm();
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);

      PARCEL_INT nstart = nstart_list_[myrank];
      PARCEL_INT nend   = nend_list_[myrank];
      PARCEL_INT nblock = nend - nstart + 1;
      //  using Tvec_type = typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type;      
      //      sycl::queue *que_ = x.queue();
      
      que_[0].wait();
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  Tvecx *send_comm_buff = (Tvecx *) send_comm_buff_[irank].data();
	  SpMV_comm_pack< Tvecx >( que_, nstart, nsend_list_[irank], nsend_gcol_parcel_list_[irank].data(), x, send_comm_buff );
	}
      }
      que_[0].wait();
      
      // --------------------------------------------------
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  send_comm_buff_[irank].getValue(send_comm_hostbuff_[irank].data());
	}
      }

      std::vector<MPI_Request> send_req(nprocess);
      std::vector<MPI_Request> recv_req(nprocess);      
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){
	  int nsend = sizeof(Tvecx) * nsend_list_[irank];
	  if(nsend_list_[irank]>0){
	    MPI_Isend( send_comm_hostbuff_[irank].data(), nsend, MPI_BYTE,irank,irank,mpi_comm,&send_req[irank]); 
	  }
	  if(nrecv_list_[irank]>0){
	    int nrecv = sizeof(Tvecx) * nrecv_list_[irank];
	    MPI_Irecv( recv_comm_hostbuff_[irank].data(), nrecv, MPI_BYTE,irank,myrank,mpi_comm,&recv_req[irank]); 
	  }	  
	}
      }

      MPI_Status istat;
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // wait
	  if(nrecv_list_[irank]>0){
	    MPI_Wait(&send_req[irank],&istat);
	  }
	  if(nsend_list_[irank]>0){
	    MPI_Wait(&recv_req[irank],&istat);
	  }
	}
      }
	              
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nrecv_list_[irank]>0) ){
#if 0
	  for(PARCEL_INT i=0;i<recv_comm_hostbuff_[irank].size();i++){
	    recv_comm_hostbuff_[irank][i] = 0.0;
	  }
#endif
	  recv_comm_buff_[irank].setValue(recv_comm_hostbuff_[irank].data());
	}
      }
      // --------------------------------------------------
      
      que_[0].wait();
      std::vector<PARCEL_INT> recv_ptr(nprocess,0);
      recv_ptr[0] = 0;
      for(PARCEL_INT irank=1;irank<nprocess;irank++){
	recv_ptr[irank] = recv_ptr[irank-1] + nrecv_list_[irank-1];
      }      
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if(myrank != irank){
	  Tvecx *recv_comm_buff = (Tvecx *) recv_comm_buff_[irank].data();
	  SpMV_comm_unpack< Tvecx >( que_, nblock, nrecv_list_[irank], recv_ptr[irank], x, recv_comm_buff );
	}
      }      
      que_[0].wait();

      return ;
      
    }
    
    template<class Ttype>
    void set_localmatrix(
			 //			 size_t n,size_t nnz,
			 PARCEL_INT nstart,PARCEL_INT nend,
			 PARCEL_INT *rowptr,
			 PARCEL_INT *crsCol,
			 Ttype *mat_mem
			 ){
      MPI_Comm mpi_comm = comm_->mpi_comm();
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);
      
      //       size_t n
      // PARCEL_INT nblock = nend - nstart;
      PARCEL_INT nblock = nend - nstart + 1;
      PARCEL_INT nodiag_ncol = 0;
      //      std::cout<<"nstart,nend,nblock = "<< nstart<<", "<<nend<<", "<<nblock<<std::endl;
#if 0
      for(PARCEL_INT j=nstart;j<=nend;j++){
#else	
      for(PARCEL_INT j=0;j<nblock;j++){
#endif	  
	for(PARCEL_INT i=rowptr[j];i<rowptr[j+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    nodiag_ncol  = nodiag_ncol + 1;
	  }
	}
      }
      
      std::vector<PARCEL_INT> globalcol_list(nodiag_ncol);
      PARCEL_INT iptr = 0;
#if 0      
      for(PARCEL_INT j=nstart;j<=nend;j++){
#else	
      for(PARCEL_INT j=0;j<nblock;j++){
#endif	
	for(PARCEL_INT i=rowptr[j];i<rowptr[j+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    globalcol_list[iptr] = crsCol[i];
	    iptr  = iptr + 1;
	  }
	}
      }

      std::sort(globalcol_list.begin(), globalcol_list.end());
      globalcol_list.erase(
			   std::unique(globalcol_list.begin(), globalcol_list.end()),
			   globalcol_list.end());
  
      std::vector<PARCEL_INT> work_int(nblock);
      std::vector<Ttype> work_type(nblock);
      
      for(PARCEL_INT irow=0;irow<nblock;irow++){

	PARCEL_INT row_ncol = rowptr[irow+1] - rowptr[irow];
	if(row_ncol>work_type.size()){
	  work_int.resize(row_ncol);
	  work_type.resize(row_ncol);
	}

	PARCEL_INT icol_ptr = 0;
	for(PARCEL_INT i = rowptr[irow];i<rowptr[irow+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (nstart<=icol) && (icol<=nend) ){
	    work_int[icol_ptr]  = crsCol[i] - nstart;
	    work_type[icol_ptr] = mat_mem[i];
	    icol_ptr = icol_ptr + 1;
	  }
	}

	PARCEL_INT igcol0 = 0;
	for(PARCEL_INT i=rowptr[irow];i<rowptr[irow+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    work_int[icol_ptr] = -1;
	    for(PARCEL_INT igcol=igcol0;igcol<globalcol_list.size();igcol++){
	      if( globalcol_list[ igcol ] == crsCol[i]){
		// work_int[icol_ptr] = nblock + 1 + igcol ;
		work_int[icol_ptr] = nblock + igcol ;
		igcol0 = igcol;
		break ;
	      }
	    }
	    work_type[icol_ptr] = mat_mem[i];
	    icol_ptr = icol_ptr + 1;
	  }
	}

	icol_ptr = 0;
	for(PARCEL_INT i=rowptr[irow];i<rowptr[irow+1];i++){
	  mat_mem[i] = work_type[icol_ptr];
	  crsCol[i]  = work_int[icol_ptr];
	  icol_ptr   = icol_ptr + 1;
	}

      }

      
      gcol_list_.resize(globalcol_list.size());
      for(PARCEL_INT i=0;i<globalcol_list.size();i++){
	gcol_list_[i] = globalcol_list[i];
      }

      // if( myrank == 1 ){
      // 	for(PARCEL_INT j=0;j<nblock;j++){
      // 	  for(PARCEL_INT i=rowptr[j];i<rowptr[j+1];i++){
      // 	    std::cout<<j<<","<<crsCol[i]<<" : "<<mat_mem[i]<<std::endl;
      // 	  }
      // 	}
      // }
      
      return ;
      
    }

    
    void set_comm_list(){
      
      MPI_Comm mpi_comm = comm_->mpi_comm();
      PARCEL_INT nstart = nstart_;
      PARCEL_INT nend   = nend_;
      
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);

      // 各プロセスの担当範囲情報を交換
      std::vector<PARCEL_INT> sbuf(nprocess);
      nstart_list_.resize(nprocess);
      nend_list_.resize(nprocess);
      for(PARCEL_INT i=0;i<nprocess;i++){sbuf[i] = nstart;}
      MPI_Alltoall(
		   sbuf.data(),1,MPI_PARCEL_INT,
		   nstart_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);
      for(PARCEL_INT i=0;i<nprocess;i++){sbuf[i] = nend;}
      MPI_Alltoall(
		   sbuf.data(),1,MPI_PARCEL_INT,
		   nend_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);
      
      // 各プロセスで必要な送信データ数情報を交換
      nsend_list_.resize(nprocess,0);
      nrecv_list_.resize(nprocess,0);
      for(PARCEL_INT i=0;i<gcol_list_.size();i++){
	PARCEL_INT icol = gcol_list_[i];
	for(PARCEL_INT irank=0;irank<nprocess;irank++){
	  if( (nstart_list_[irank]<=icol) && (icol<=nend_list_[irank]) ){
	    nrecv_list_[irank] = nrecv_list_[irank] +1;
	  }
	}
      }  
      MPI_Alltoall(
		   nrecv_list_.data(),1,MPI_PARCEL_INT,
		   nsend_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);
  
      // 送信してほしい列番号のベクトルのリストを交換
      nrecv_gcol_list_.resize(nprocess);
      nsend_gcol_list_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if(irank != myrank){
	  PARCEL_INT recv_nbuff = nrecv_list_[irank];
	  if( recv_nbuff > 0 ){
	    nrecv_gcol_list_[irank].resize(recv_nbuff);
	  }
	  PARCEL_INT send_nbuff = nsend_list_[irank];
	  if( send_nbuff > 0 ){
	    nsend_gcol_list_[irank].resize(send_nbuff);
	  }
	}
      }

      std::vector<PARCEL_INT> gcol_list_ptr(nprocess,0);
      for(PARCEL_INT i=0;i<gcol_list_.size();i++){
	PARCEL_INT icol = gcol_list_[i];
	for(PARCEL_INT irank=0;irank<nprocess;irank++){
	  if( (nstart_list_[irank]<=icol) && (icol<=nend_list_[irank]) ){
	    PARCEL_INT list_ptr = gcol_list_ptr[irank];	    
	    nrecv_gcol_list_[irank][list_ptr] = icol;
	    gcol_list_ptr[irank] = gcol_list_ptr[irank] + 1;
	  }
	}
      }

      std::vector<MPI_Request> send_req(nprocess);
      std::vector<MPI_Request> recv_req(nprocess);      
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // send
	  if(nrecv_list_[irank]>0){	    
	    MPI_Isend( nrecv_gcol_list_[irank].data(), nrecv_list_[irank],MPI_PARCEL_INT,irank,myrank,mpi_comm,&send_req[irank]); 
	  }
	  // recv
	  if(nsend_list_[irank]>0){
	    MPI_Irecv( nsend_gcol_list_[irank].data(), nsend_list_[irank],MPI_PARCEL_INT,irank,irank,mpi_comm,&recv_req[irank]); 
	  }	  
	}
      }
      
      MPI_Status istat;
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // wait
	  if(nrecv_list_[irank]>0){
	    MPI_Wait(&send_req[irank],&istat);
	  }
	  if(nsend_list_[irank]>0){
	    MPI_Wait(&recv_req[irank],&istat);
	  }
	}
      }
      MPI_Barrier(mpi_comm);
      
#if 1
      // host用MPIバッファ
      send_comm_hostbuff_.resize(nprocess);
      recv_comm_hostbuff_.resize(nprocess);

      // SYCL用MPIバッファ
      send_comm_buff_.resize(nprocess);
      recv_comm_buff_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  send_comm_buff_[irank].resize(nsend_list_[irank]);
	  send_comm_hostbuff_[irank].resize(nsend_list_[irank]);
	}
	if( (myrank != irank) && (nrecv_list_[irank]>0) ){
	  recv_comm_buff_[irank].resize(nrecv_list_[irank]);
	  recv_comm_hostbuff_[irank].resize(nrecv_list_[irank]);
	}
      }
      
      nsend_gcol_parcel_list_.resize(nprocess);
      nrecv_gcol_parcel_list_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank] > 0) ){
	  nsend_gcol_parcel_list_[irank].resize( nsend_list_[irank] );
	  nsend_gcol_parcel_list_[irank].setValue( nsend_gcol_list_[irank].data() );
	}
	if( (myrank != irank) && (nrecv_list_[irank] > 0) ){
	  nrecv_gcol_parcel_list_[irank].resize( nrecv_list_[irank] );
	  nrecv_gcol_parcel_list_[irank].setValue( nrecv_gcol_list_[irank].data() );
	}
      }
      
#endif
      
      return ;
      
    }
    
    PARCEL_INT get_send_size(){
      PARCEL_INT nvec_size = 0;
      for(PARCEL_INT i=0;i<nsend_list_.size();i++){
	if( nsend_list_[i] > 0){ 
	  nvec_size = nvec_size + nsend_list_[i];
	}
      }
      return nvec_size;
    }

    PARCEL_INT get_recv_size(){
      PARCEL_INT nvec_size = 0;
      for(PARCEL_INT i=0;i<nrecv_list_.size();i++){
	if( nrecv_list_[i] > 0){ 
	  nvec_size = nvec_size + nrecv_list_[i];
	}
      }
      return nvec_size;
    }
    
    void set_comm(parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend){
      comm_ = comm;
      nstart_ = nstart;
      nend_   = nend;
      size_t nlocal = nend - nstart + 1;

      int myrank   = comm[0].mpi_rank();
      int nprocess = comm[0].mpi_size();
      
#if 0
      PARCEL_INT buffsize = 1;
      std::vector<PARCEL_INT> sbuff(nprocess,0);
      nstart_all_.resize(nprocess);
      nend_all_.resize(nprocess);
      
      for(size_t i=0;i<nprocess;i++){sbuff[i] = nstart;}
      comm_[0].alltoall<PARCEL_INT>(sbuff,1,nstart_all_,1);
      
      for(size_t i=0;i<nprocess;i++){sbuff[i] = nend;}
      comm_[0].alltoall<PARCEL_INT>(sbuff,1,nend_all_  ,1);
      
      gn_ = 0;
      for(size_t i=0;i<nprocess;i++){
	gn_ = gn_ + nend_all_[i] - nstart_all_[i] + 1;
      }
      
#endif
      //      set_comm_list(comm->mpi_comm(),nstart_,nend_);
  

#if 0
      if( myrank == 0){
	for(size_t i=0;i<nprocess;i++){
	  std::cout<<" rank "<<i<<" : "<<nstart_all_[i]<<","<<nend_all_[i]<<std::endl;
	}
      }
#endif
      
      return ;      
      
    }
    
    parcel::comm *comm(){
      return comm_;
    }    
    
#endif
    
    void resize(size_t n,size_t nnz, sycl::queue* q){
      q_ = q;
      n_   = n>1? n:1;
      nnz_ = nnz>1? nnz:1;
      mat_.resize(nnz_ , q_);
      rowptr_.resize(n_+1 , q_);
      crsCol_.resize(nnz_ , q_);
      maxCol_ = 0;
    }

    crsMat(){}
    crsMat(size_t n, size_t nnz,sycl::queue* q){
       resize(n,nnz,q);
    }
    ~crsMat(){
      //      q_[0].wait();
    }

    sycl::queue *queue(){
      return q_;
    }

    auto mat_data(){
      return mat_.data();
    }

    vector<parceltype> &mat(){
      return mat_;
    }

    auto mat_data_hi(){
      return mat_.data_hi();
    }

    auto mat_data_lo(){
      return mat_.data_lo();
    }

    template<class Ttype>
    void mat_setValue(Ttype *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem_hi,double *mem_lo){
      mat_.setValue(mem_hi,mem_lo);
    }

    void mat_setValue_hi(double *mem){
      mat_.setValue_hi(mem);
    }
    void mat_setValue_lo(double *mem){
      mat_.setValue_lo(mem);
    }

    template<class Ttype>
    void mat_getValue(Ttype *mem){
      mat_.getValue(mem);
    }

    void mat_getValue(double *mem){
      mat_.getValue(mem);
    }

    void mat_getValue(double *mem_hi,double *mem_lo){
      mat_.getValue(mem_hi,mem_lo);
    }

    void mat_getValue_hi(double *mem){
      mat_.getValue_hi(mem);
    }
    void mat_getValue_lo(double *mem){
      mat_.getValue_lo(mem);
    }

    util_sycl::vector<PARCEL_INT> *rowptr(){
      return &rowptr_;
    }

    PARCEL_INT *rowptr_data(){
      return rowptr_.data();
    }

    util_sycl::vector<PARCEL_INT> *crsCol(){
      return &crsCol_;
    }

    PARCEL_INT *crsCol_data(){
      return crsCol_.data();
    }

    void rowptr_setValue(PARCEL_INT *mem){
      rowptr_.setValue(mem);
    }
    void crsCol_setValue(PARCEL_INT *mem){
      crsCol_.setValue(mem);
      size_t maxCol = 0;
    }

    void rowptr_getValue(PARCEL_INT *mem){
      rowptr_.getValue(mem);
    }
    void crsCol_getValue(PARCEL_INT *mem){
      crsCol_.getValue(mem);
      size_t maxCol = 0;
    }

    void set_maxCol(){

      q_[0].wait();
      size_t MAX   = n_;
#if _A64FX_HIPSYCL_
      const size_t nblock      = 12;
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = n_;
      const size_t nlocalblock = 1;
#endif

      util_sycl::vector<PARCEL_INT> nmaxCol_all_;
      nmaxCol_all_.resize(nblock);
      PARCEL_INT* nmaxCol_all = nmaxCol_all_.data();
      sycl::buffer<PARCEL_INT> buff(nmaxCol_all,nblock);
      PARCEL_INT* Rowptr = rowptr_.data();
      PARCEL_INT N = n_;
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  size_t nmaxCol = 0;
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){
	      size_t nCol = Rowptr[i+1] - Rowptr[i];
	      nmaxCol = nmaxCol < nCol ? nCol :nmaxCol;
	    }
	  }
	  nmaxCol_all[ii] = nmaxCol;
	});
      q_[0].wait();
      std::vector<PARCEL_INT> nmaxCol_all_host(nblock,0);
      nmaxCol_all_.getValue(nmaxCol_all_host.data(),nblock);
      size_t nmaxCol = 0;
      for(size_t i=0;i<nblock;i++){
	size_t nCol = nmaxCol_all_host[i];
	nmaxCol = nmaxCol < nCol ? nCol :nmaxCol;
      }
      maxCol_ = nmaxCol;
    }

    size_t maxCol(){
      return maxCol_;
    }

    size_t nnz(){
      return nnz_;
    }

    size_t n(){
      return n_;
    }

  };

  template<class parceltype>
  class diaMat{
    //    parceltype mat_;
    vector<parceltype> mat_;
    util_sycl::vector<PARCEL_INT> offset_;
    PARCEL_INT max_up_offset_   ;
    PARCEL_INT max_down_offset_ ;
    size_t n_;
    size_t nnd_;
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};
  public:

    void set_queue(sycl::queue* q){
      q_ = q;
      return ;    
    }

    void resize(size_t n,size_t nnd, sycl::queue *q){
      q_   = q;
      n_   = n>1? n:1;
      nnd_ = nnd>1? nnd:1;

      mat_.resize(n_*nnd_ , q_);
      offset_.resize(nnd_ ,q_);
      // max_up_offset_ = n;
      // max_down_offset_ = n; 
    }

    diaMat(){}
    diaMat(size_t n, size_t nnd,sycl::queue* q){
       resize(n,nnd,q);
    }
    ~diaMat(){
      //      q_[0].wait();
    }

    sycl::queue *queue(){
      return q_;
    }

    template<class Ttype>
    void mat_setValue(Ttype *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem_hi,double *mem_lo){
      mat_.setValue(mem_hi,mem_lo);
    }

    void mat_setValue_hi(double *mem){
      mat_.setValue_hi(mem);
    }
    void mat_setValue_lo(double *mem){
      mat_.setValue_lo(mem);
    }

    vector<parceltype> &mat(){
      return mat_;
    }

    auto mat_data(){
      return mat_.data();
    }

    auto mat_data_hi(){
      return mat_.data_hi();
    }

    auto mat_data_lo(){
      return mat_.data_lo();
    }

    PARCEL_INT *offset_data(){
      return offset_.data();
    }

    void offset_setValue(PARCEL_INT *mem){
      PARCEL_INT max_up_offset  = 0;
      PARCEL_INT max_down_offset= 0;
      for(size_t i = 0; i<nnd_;i++){
	// std::cout<<"offset = "<<mem[i]<<std::endl;
	max_up_offset   = max_up_offset   > mem[i] ? mem[i] : max_up_offset;
	max_down_offset = max_down_offset < mem[i] ? mem[i] : max_down_offset;
      }
       // std::cout<<"max_up_offset   = "<<max_up_offset<<std::endl;
       // std::cout<<"max_down_offset = "<<max_down_offset<<std::endl;
      max_up_offset_   = max_up_offset;
      max_down_offset_ = max_down_offset;
      offset_.setValue(mem);
    }

    util_sycl::vector<PARCEL_INT> &offset(){
      return offset_;
    };

    size_t nnd(){
      return nnd_;
    }

    size_t n(){
      return n_;
    }

    void set_max_up_offset(PARCEL_INT max_up_offset){
      max_up_offset_ = max_up_offset;
      return ;
    }
    void set_max_down_offset(PARCEL_INT max_down_offset){
      max_down_offset_ = max_down_offset;
      return ;
    }

    PARCEL_INT max_up_offset(){
      return max_up_offset_;
    }
    PARCEL_INT max_down_offset(){
      return max_down_offset_;
    }

  };

  template<class parceltype>
  class matrix{
    size_t matrix_type_;
    crsMat<parceltype> crsMat_;
    diaMat<parceltype> diaMat_;
    sycl::queue* q_;
  public:

    matrix(sycl::queue* q){
      q_ = q;
      matrix_type_ = -1;
    }
    matrix(){}
    ~matrix(){}

    crsMat<parceltype> *crsMat_data(){
      return &crsMat_;
    }

#if PARCEL_MPI
    
    parcel::comm *comm(){
      if(matrix_type_==CRS_MATRIX){
	return crsMat_.comm();
      }
      if(matrix_type_==DIA_MATRIX){
	// diaMat_.comm();
	// return crsMat_.comm();
      }
      // return crsMat_.comm();
    }
    
    PARCEL_INT get_recv_size(){
      if(matrix_type_==CRS_MATRIX){
	return crsMat_.get_recv_size();
      }
      if(matrix_type_==DIA_MATRIX){
	// return diaMat_.get_recv_size();
	//return crsMat_.get_recv_size();
      }
      // return crsMat_.get_recv_size();
    }
    
#endif    
    
    void resize(sycl::queue* q,size_t n,size_t nnz_nnd,size_t matrix_type){
      set_queue(q);
      matrix_type_ = matrix_type;
      if(matrix_type_==CRS_MATRIX){
	crsMat_.resize(n,nnz_nnd,q_);
      }
      if(matrix_type_==DIA_MATRIX){
	diaMat_.resize(n,nnz_nnd,q_);
      }
      return ;
    }

    void resize(size_t n,size_t nnz_nnd,size_t matrix_type){
      matrix_type_ = matrix_type;
      if(matrix_type_==CRS_MATRIX){
	crsMat_.resize(n,nnz_nnd,q_);
      }
      if(matrix_type_==DIA_MATRIX){
	diaMat_.resize(n,nnz_nnd,q_);
      }
      return ;
    }

    // 入力mat_memはdouble型に固定
    template<class Ttype>
    void set_crsMat(
#if PARCEL_MPI
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
#endif		    
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    Ttype *mat_mem){
#if PARCEL_MPI
      
      //      std::cout<<"PARCEL MPI START"<<std::endl;
      // MPIコミュニケータの設定 
      //      std::cout<<"     set_comm"<<std::endl;
      crsMat_.set_comm(comm,nstart,nend);

      //      std::cout<<"     set_localmatrix"<<std::endl;
      // 分散行列用にCRS行列を変換する
      crsMat_.set_localmatrix(
			      nstart,nend,
			      rowptr_mem,
			      crsCol_mem,
			      mat_mem
			      );
      
      //      std::cout<<"     set_comm_list"<<std::endl;
      // 通信テーブルを作成
      crsMat_.set_comm_list();
      //      std::cout<<"PARCEL MPI END"<<std::endl;
#endif
      
      crsMat_.resize(n,nnz,q_);
      crsMat_.mat_setValue(mat_mem);
      crsMat_.rowptr_setValue(rowptr_mem);
      crsMat_.crsCol_setValue(crsCol_mem);
      // 最大列数の取得
      crsMat_.set_maxCol();
      // std::cout<<"maxCol = "<<crsMat_.maxCol()<<std::endl;
      matrix_type_ = CRS_MATRIX;
      return ;
    }
    
    void set_crsMat(
#if PARCEL_MPI
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
#endif		    
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    double *mat_mem_hi,double *mat_mem_lo
		    ){
      crsMat_.resize(n,nnz,q_);
      crsMat_.mat_setValue(mat_mem_hi,mat_mem_lo);
      crsMat_.rowptr_setValue(rowptr_mem);
      crsMat_.crsCol_setValue(crsCol_mem);

#if PARCEL_MPI
      
#if 0
      // MPIコミュニケータの設定 
      crsMat_.set_comm(comm,nstart,nend);

      // 分散行列用にCRS行列を変換する
      // ここはdd形式に対応させる
      crsMat_.set_localmatrix(
			      nstart,nend,
			      rowptr_mem,
			      crsCol_mem,
			      mat_mem
			      );
      // 通信テーブルを作成
      crsMat_.set_comm_list();
#endif
      
#endif      
      matrix_type_ = CRS_MATRIX;
      return ;
    }

    // -------
    // DIA Matrix
    diaMat<parceltype> *diaMat_data(){
      return &diaMat_;
    }
    template<class Ttype>
    void set_diaMat(
		    size_t n,size_t nnd, 
		    PARCEL_INT *offset_mem,
		    Ttype *mat_mem){
      diaMat_.resize(n,nnd,q_);
      diaMat_.mat_setValue(mat_mem);
      diaMat_.offset_setValue(offset_mem);
      matrix_type_ = DIA_MATRIX;
      return ;
    }

    void set_diaMat(
		    size_t n,size_t nnd, 
		    PARCEL_INT *offset_mem,
		    double *mat_mem_hi,double *mat_mem_lo){
      diaMat_.resize(n,nnd,q_);
      diaMat_.mat_setValue(mat_mem_hi,mat_mem_lo);
      diaMat_.offset_setValue(offset_mem);
      matrix_type_ = DIA_MATRIX;
      return ;
    }

    void set_queue(sycl::queue* q){
      q_ = q;
      return ;
    }

    sycl::queue *queue(){
      return q_;
    }

    size_t matrix_type(){
      return matrix_type_;
    }

  };


#if 0
  template<class T=double>
  class scalar{
    T mem_;
    ddlib::ddtype mem_dd_;
  public:
    scalar(){
      mem_ = 0.0;
      mem_dd_.set_dd(0.0,0.0);
    }
    ~scalar(){}

    T get_T(){
      return mem_;
    }
    ddlib::ddtype get_dd(){
      return mem_dd_;
    }

    void set_T(T mem){
      mem_ = mem;
    }
    void set_dd(ddlib::ddtype mem_dd){
      mem_dd_ = mem_dd;
    }
  };
#endif

}


// #include "parcel_krylov.hpp"
// #include "parcel_blas.hpp"

#endif
