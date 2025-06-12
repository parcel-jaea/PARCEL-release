#ifndef PARCEL_PRECON_BLOCKJACOBI_
#define PARCEL_PRECON_BLOCKJACOBI_

namespace parcel{  

  template<
    class Tmat,
    class Tprecon_mat,
    class Tprecon_vec
    >
  class blockjacobi_buffer {
    // blockjacobi前処理
    //    util_sycl::vector<size_t> n_;
    util_sycl::vector<size_t> nhead_;
    util_sycl::vector<size_t> ntail_;
    util_sycl::vector<size_t> nnzL_;
    util_sycl::vector<size_t> nnzU_;

    size_t nnzL_size_ = 0;
    size_t nnzU_size_ = 0;
    size_t nndL_size_ = 0;
    size_t nndU_size_ = 0;
    matrix<Tprecon_mat> Lmatrix_;
    matrix<Tprecon_mat> Umatrix_;
    vector<Tprecon_vec> work_;

  public:
    blockjacobi_buffer(){}
    ~blockjacobi_buffer(){}

    void resize_blocksize(
			  sycl::queue *q,
			  size_t numblock
			  ){
      //      n_.resize(numblock,q);

      nhead_.resize(numblock,q);
      ntail_.resize(numblock,q);
      nnzL_.resize(numblock,q);
      nnzU_.resize(numblock,q);
    }

    void resize_buffer(
		       parcel::crsMat<Tmat> &crsMat
		       ){
      size_t matrix_type = CRS_MATRIX;
      sycl::queue *q = crsMat.queue();
      const size_t n = crsMat.n();
      Lmatrix_.resize(q,n,nnzL_size_,matrix_type);
      Umatrix_.resize(q,n,nnzU_size_,matrix_type);
#if PARCEL_MPI
      PARCEL_INT nadd = crsMat.get_recv_size();
      work_.resize(n+nadd,q);
#else
      work_.resize(n,q);
#endif
    }

    void resize_buffer(
		       parcel::diaMat<Tmat> &diaMat
		       ){
      size_t matrix_type = DIA_MATRIX;
      sycl::queue *q = diaMat.queue();
      const size_t n = diaMat.n();
      Lmatrix_.resize(q,n,nndL_size_,matrix_type);
      Umatrix_.resize(q,n,nndU_size_,matrix_type);
      work_.resize(n,q);
    }

    // void n_setValue(size_t *mem){
    //   n_.setValue(mem);
    //   return ;
    // }
    void nhead_setValue(size_t *mem){
      nhead_.setValue(mem);
      return ;
    }
    void ntail_setValue(size_t *mem){
      ntail_.setValue(mem);
      return ;
    }
    void nnzL_setValue(size_t *mem){
      nnzL_.setValue(mem);
      return ;
    }
    void nnzU_setValue(size_t *mem){
      nnzU_.setValue(mem);
      return ;
    }
    // void n_getValue(size_t *mem){
    //   n_.getValue(mem);
    //   return ;
    // }
    void nhead_getValue(size_t *mem){
      nhead_.getValue(mem);
      return ;
    }
    void ntail_getValue(size_t *mem){
      ntail_.getValue(mem);
      return ;
    }
    void nnzL_getValue(size_t *mem){
      nnzL_.getValue(mem);
      return ;
    }
    void nnzU_getValue(size_t *mem){
      nnzU_.getValue(mem);
      return ;
    }

    // void @_setValue(size_t *mem){
    //   @_.setValue(mem);
    //   return ;
    // }

    void set_nnzL_size(size_t nnzL_size){
      nnzL_size_ = nnzL_size;
    }
    void set_nnzU_size(size_t nnzU_size){
      nnzU_size_ = nnzU_size;
    }
    size_t get_nnzL_size(){
      return nnzL_size_;
    }
    size_t get_nnzU_size(){
      return nnzU_size_;
    }

    void set_nndL_size(size_t nndL_size){
      nndL_size_ = nndL_size;
    }
    void set_nndU_size(size_t nndU_size){
      nndU_size_ = nndU_size;
    }
    size_t get_nndL_size(){
      return nndL_size_;
    }
    size_t get_nndU_size(){
      return nndU_size_;
    }

    //    size_t *n_data(){return n_.data();}      
    size_t *nhead_data(){return nhead_.data();}
    size_t *ntail_data(){return ntail_.data();}
    size_t *nnzL_data(){return nnzL_.data();}
    size_t *nnzU_data(){return nnzU_.data();}
    util_sycl::vector<size_t> &nhead(){return nhead_;}
    util_sycl::vector<size_t> &ntail(){return ntail_;}

    matrix<Tprecon_mat> &Lmatrix(){return Lmatrix_;}
    matrix<Tprecon_mat> &Umatrix(){return Umatrix_;}
    vector<Tprecon_vec> &work(){return work_;}

  };
  
  template <
    class Tmat,
    class Tvec,
    class Tprecon_mat,
    class Tprecon_vec,
    class Tprecon_calc_mat,
    class Tcalc
    >
  class precon_blockjacobi : public mixed_precison::mixed_precison_blas{
    size_t n_ = 1;
    size_t nblock_ = 1;
    //    PARCEL_INT BLOCK_JACOBI_NUMBLOCK_ = 4;
    PARCEL_INT BLOCK_JACOBI_NUMBLOCK_ = 1; // ブロックヤコビの分割数
    blockjacobi_buffer<Tmat,Tprecon_mat,Tprecon_vec> blockjacobi_buffer_;

    bool SCALING_MATRIX_ = false;
    bool SCALING_VECTOR_ = false; 
    vector<parcel_double> scalingInv_vec_;

  public:
    PARCEL_INT numblock(){
      return BLOCK_JACOBI_NUMBLOCK_;
    }
    precon_blockjacobi(){}
    ~precon_blockjacobi(){}

    void set_SCALING_MATRIX(bool scaling){
      SCALING_MATRIX_ = scaling;
      return ;
    }
    void set_SCALING_VECTOR(bool scaling){
      SCALING_VECTOR_ = scaling;
      return ;
    }
 
    void resize_blocksize(
			  crsMat<Tmat> &crsMat,
			  PARCEL_INT BLOCK_JACOBI_NUMBLOCK){
      sycl::queue *q_ = crsMat.queue();
      n_ = crsMat.n();
      BLOCK_JACOBI_NUMBLOCK_ = BLOCK_JACOBI_NUMBLOCK;
      size_t nblock = n_ / BLOCK_JACOBI_NUMBLOCK_ + 1;
      nblock_ = nblock;
      blockjacobi_buffer_.resize_blocksize(q_,BLOCK_JACOBI_NUMBLOCK_);
      return ;      
    }
 
    void resize_blocksize(
			  diaMat<Tmat> &diaMat,
			  PARCEL_INT BLOCK_JACOBI_NUMBLOCK){
      sycl::queue *q_ = diaMat.queue();
      n_ = diaMat.n();
      BLOCK_JACOBI_NUMBLOCK_ = BLOCK_JACOBI_NUMBLOCK;
      size_t nblock = n_ / BLOCK_JACOBI_NUMBLOCK_ + 1;
      nblock_ = nblock;
      blockjacobi_buffer_.resize_blocksize(q_,BLOCK_JACOBI_NUMBLOCK_);
      return ;      
    }

    void resize_buffer(crsMat<Tmat> &crsMat){
      blockjacobi_buffer_.resize_buffer(crsMat);
      return ;
    }

    void resize_buffer(diaMat<Tmat> &diaMat){
      blockjacobi_buffer_.resize_buffer(diaMat);
      return ;
    }

    void get_blockjacobi_nnzLnnzU(
				  parcel::crsMat<Tmat> &crsMat
				  ){
      sycl::queue *q_ = crsMat.queue();
      size_t N   = crsMat.n();
      size_t NNZ = crsMat.nnz();

      // block数と各blockのサイズを計算 
      size_t addL  = N%BLOCK_JACOBI_NUMBLOCK_ ;
      size_t nhead = 0;
      size_t ntail = 0;
      size_t Nblock = N / BLOCK_JACOBI_NUMBLOCK_;

      //      std::vector<size_t> n_block(BLOCK_JACOBI_NUMBLOCK_,0);
      std::vector<size_t> nhead_block(BLOCK_JACOBI_NUMBLOCK_,0);
      std::vector<size_t> ntail_block(BLOCK_JACOBI_NUMBLOCK_,0);

      for(PARCEL_INT id = 0;id<BLOCK_JACOBI_NUMBLOCK_;id++){
	nhead = ntail;
	ntail = ntail + Nblock;
	if(addL>id){
	  ntail = ntail + 1;
	}
	nhead_block[id] = nhead;
	ntail_block[id] = ntail;
	// std::cout<<id<<" : head,tail = "<<nhead<<" --> "<<ntail<<std::endl;
      }

      blockjacobi_buffer_.nhead_setValue(nhead_block.data());
      blockjacobi_buffer_.ntail_setValue(ntail_block.data());

      vector<Tmat> &mat=crsMat.mat();
      PARCEL_INT *Rowptr = crsMat.rowptr_data();
      PARCEL_INT *crsCol = crsMat.crsCol_data();

      const size_t work_group_size = 32; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
      
      size_t *nhead_ = blockjacobi_buffer_.nhead_data();
      size_t *ntail_ = blockjacobi_buffer_.ntail_data();
      size_t *nnzL_ = blockjacobi_buffer_.nnzL_data();
      size_t *nnzU_ = blockjacobi_buffer_.nnzU_data();

      util_sycl::vector<size_t> nnzL_buf;
      util_sycl::vector<size_t> nnzU_buf;

      nnzL_buf.resize(num_work_items,q_);
      nnzU_buf.resize(num_work_items,q_);
      size_t *nnzL_ptr = nnzL_buf.data();
      size_t *nnzU_ptr = nnzU_buf.data();

      q_[0].wait();
#if _A64FX_HIPSYCL_

      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
			   size_t *nnzL = &nnzL_ptr[group_id*work_group_size];
			   size_t *nnzU = &nnzU_ptr[group_id*work_group_size];
			   nnzL[local_id] = 0;
			   nnzU[local_id] = 0;

			   //			   printf("local_id = %u \n",local_id);			   
			   size_t ihead = nhead_[group_id];
			   size_t itail = ntail_[group_id];
			   for(size_t i = ihead;i<itail;i++){
			     // for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
			     for(PARCEL_INT j = Rowptr[i] + local_id; j < Rowptr[i+1]; j=j+work_group_size){
			       const PARCEL_INT col = crsCol[j];
			       if( (col<i) && (ihead<=col) ){
				 nnzL[local_id] = nnzL[local_id] + 1;
				 continue ;
			       }
			       if( (col>i) && (col<itail) ){
				 nnzU[local_id] = nnzU[local_id] + 1;
				 continue ;
			       }
			     }
			   }

			   // Do a tree reduction on items in work-group
			   for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			     item.barrier(sycl::access::fence_space::local_space);
			     if (local_id < i){
			       nnzL[local_id] += nnzL[local_id + i];
			       nnzU[local_id] += nnzU[local_id + i];
			     }
			   } 

			   if (local_id == 0) {
			     //			     printf("id = %d : nnzL,nnzU : %d,%d,%d \n",group_id,nnzL[0],nnzU[0],n_[group_id]);
			     // outbuf[group_id] = scratch[0];
			     size_t n_size = itail - ihead; // 対角要素数
			     nnzL_[group_id] = nnzL[0] + n_size;
			     nnzU_[group_id] = nnzU[0] + n_size;
			   }

#else
      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  size_t global_id = ii;

	  size_t nnzL = 0;
	  size_t nnzU = 0;

	  size_t ihead = nhead_[global_id];
	  size_t itail = ntail_[global_id];
	  for(size_t i = ihead;i<itail;i++){
	    for(PARCEL_INT j = Rowptr[i] ; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      if( (col<i) && (ihead<=col) ){
		nnzL = nnzL + 1;
		continue ;
	      }
	      if( (col>i) && (col<itail) ){
		nnzU = nnzU + 1;
		continue ;
	      }
	    }
	  }
	  
	  size_t n_size = itail - ihead; // 対角要素数
	  nnzL_[global_id] = nnzL + n_size;
	  nnzU_[global_id] = nnzU + n_size;	  
#endif


// #if _A64FX_HIPSYCL_
#if _A64FX_HIPSYCL_
			 });
#else

#endif
			 
});
      q_[0].wait();

      std::vector<size_t> nnzL_block(BLOCK_JACOBI_NUMBLOCK_,99);
      std::vector<size_t> nnzU_block(BLOCK_JACOBI_NUMBLOCK_,99);

      blockjacobi_buffer_.nnzL_getValue(nnzL_block.data());
      blockjacobi_buffer_.nnzU_getValue(nnzU_block.data());
      q_[0].wait();

      size_t nnzL_sum = 0;
      size_t nnzU_sum = 0;
      for(PARCEL_INT id = 0;id<BLOCK_JACOBI_NUMBLOCK_;id++){
	nnzL_sum = nnzL_sum + nnzL_block[id];
	nnzU_sum = nnzU_sum + nnzU_block[id];
	//	std::cout<<"nnz id : "<<id<<" : "<<nnzL_block[id]<<" , "<<nnzU_block[id]<<std::endl;
      }
      //      std::cout<<"set : nnzL_sum = "<<nnzL_sum<<std::endl;
      //      std::cout<<"set : nnzU_sum = "<<nnzU_sum<<std::endl;

      blockjacobi_buffer_.set_nnzL_size(nnzL_sum);
      blockjacobi_buffer_.set_nnzU_size(nnzU_sum);

      std::vector<size_t> nnzL_block_ptr(BLOCK_JACOBI_NUMBLOCK_,99);
      std::vector<size_t> nnzU_block_ptr(BLOCK_JACOBI_NUMBLOCK_,99);
      nnzL_block_ptr[0] = 0;
      nnzU_block_ptr[0] = 0;
      for(PARCEL_INT id = 1;id<BLOCK_JACOBI_NUMBLOCK_;id++){
	nnzL_block_ptr[id] = nnzL_block_ptr[id-1] + nnzL_block[id-1];
	nnzU_block_ptr[id] = nnzU_block_ptr[id-1] + nnzU_block[id-1];
      }

      blockjacobi_buffer_.nnzL_setValue(nnzL_block_ptr.data());
      blockjacobi_buffer_.nnzU_setValue(nnzU_block_ptr.data());
      q_[0].wait();

      return ;

    }

    void get_blockjacobi_nndLnndU(
				  parcel::diaMat<Tmat> &diaMat
				  ){
      sycl::queue *q_ = diaMat.queue();
      size_t N   = diaMat.n();
      size_t NND = diaMat.nnd();
      const PARCEL_INT max_up_offset   = diaMat.max_up_offset();
      const PARCEL_INT max_down_offset = diaMat.max_down_offset();

      // block数と各blockのサイズを計算 
      size_t addL  = N%BLOCK_JACOBI_NUMBLOCK_ ;
      size_t nhead = 0;
      size_t ntail = 0;
      size_t Nblock = N / BLOCK_JACOBI_NUMBLOCK_;

      //      std::vector<size_t> n_block(BLOCK_JACOBI_NUMBLOCK_,0);
      std::vector<size_t> nhead_block(BLOCK_JACOBI_NUMBLOCK_,0);
      std::vector<size_t> ntail_block(BLOCK_JACOBI_NUMBLOCK_,0);

      for(PARCEL_INT id = 0;id<BLOCK_JACOBI_NUMBLOCK_;id++){
	nhead = ntail;
	ntail = ntail + Nblock;
	if(addL>id){
	  ntail = ntail + 1;
	}
	nhead_block[id] = nhead;
	ntail_block[id] = ntail;
	// std::cout<<id<<" : head,tail = "<<nhead<<" --> "<<ntail<<std::endl;
      }

      blockjacobi_buffer_.nhead_setValue(nhead_block.data());
      blockjacobi_buffer_.ntail_setValue(ntail_block.data());

      q_[0].wait();

#if 0
      nhead_block[0] = 111;
      ntail_block[0] = 222;
      q_[0].wait();
      std::cout<<"@ 1 @ : head,tail = "<<nhead_block[0]<<" --> "<<ntail_block[0]<<std::endl;
      blockjacobi_buffer_.nhead_getValue(nhead_block.data());
      blockjacobi_buffer_.ntail_getValue(ntail_block.data());
      q_[0].wait();
      std::cout<<"@ 2 @ : head,tail = "<<nhead_block[0]<<" --> "<<ntail_block[0]<<std::endl;
#endif

#if 0
      std::vector<size_t> nhead_cpu(BLOCK_JACOBI_NUMBLOCK_,-1);
      std::vector<size_t> ntail_cpu(BLOCK_JACOBI_NUMBLOCK_,-1);
      nhead_cpu[0] = 333;
      ntail_cpu[0] = 444;
      q_[0].wait();
      std::cout<<"@ 3 @ : head,tail = "<<nhead_cpu[0]<<" --> "<<ntail_cpu[0]<<std::endl;
      blockjacobi_buffer_.nhead_getValue(nhead_cpu.data());
      blockjacobi_buffer_.ntail_getValue(ntail_cpu.data());
      q_[0].wait();
      std::cout<<"@ 4 @ : head,tail = "<<nhead_cpu[0]<<" --> "<<ntail_cpu[0]<<std::endl;
#endif

      q_[0].wait();


      vector<Tmat> &mat=diaMat.mat();
      PARCEL_INT *offset = diaMat.offset_data();

      const size_t work_group_size = 32; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
      //      size_t *nhead_ = blockjacobi_buffer_.nhead_data();
      //      size_t *ntail_ = blockjacobi_buffer_.ntail_data();

      // offsetを取得してホスト側で処理する
      util_sycl::vector<PARCEL_INT> &offset_vec = diaMat.offset();
      std::vector<PARCEL_INT> offset_cpu(NND,0);
      std::vector<PARCEL_INT> offsetL_cpu(NND,0);
      std::vector<PARCEL_INT> offsetU_cpu(NND,0);
      offset_vec.getValue( offset_cpu.data() );

      size_t nndL = 0;
      size_t nndU = 0;

      offsetU_cpu[nndU] = 0;
      nndU = nndU + 1;
      for(int i=0;i<NND;i++){
	if(offset_cpu[i]<0){
	  offsetL_cpu[nndL] = offset_cpu[i];
	  nndL = nndL + 1;
	}
	if(0<offset_cpu[i]){
	  offsetU_cpu[nndU] = offset_cpu[i];
	  nndU = nndU + 1;
	}
      }
      offsetL_cpu[nndL] = 0;
      nndL = nndL + 1;

      blockjacobi_buffer_.set_nndL_size(nndL);
      blockjacobi_buffer_.set_nndU_size(nndU);

      resize_buffer(diaMat);

#if 1
      matrix<Tprecon_mat> &Lmatrix_matrix = blockjacobi_buffer_.Lmatrix();
      parcel::diaMat<Tprecon_mat> &Lmatrix_diaMat = *Lmatrix_matrix.diaMat_data();
      util_sycl::vector<PARCEL_INT> &Lmatrix_offset_vec = Lmatrix_diaMat.offset();
      Lmatrix_offset_vec.setValue( offsetL_cpu.data() );
      Lmatrix_diaMat.set_max_up_offset(max_up_offset);
      Lmatrix_diaMat.set_max_down_offset(0);

      matrix<Tprecon_mat> &Umatrix_matrix = blockjacobi_buffer_.Umatrix();
      parcel::diaMat<Tprecon_mat> &Umatrix_diaMat = *Umatrix_matrix.diaMat_data();
      util_sycl::vector<PARCEL_INT> &Umatrix_offset_vec = Umatrix_diaMat.offset();
      Umatrix_offset_vec.setValue( offsetU_cpu.data() );

      Umatrix_diaMat.set_max_up_offset(0);
      Umatrix_diaMat.set_max_down_offset(max_down_offset);

#endif

      q_[0].wait();

      return ;

    }

    void get_blockjacobi_LUmatrix(
				  parcel::crsMat<Tmat> &crsMat
				  ){

      sycl::queue *q_ = crsMat.queue();
      setQueue(q_);
      size_t N   = crsMat.n();
      size_t NNZ = crsMat.nnz();

      vector<Tmat> &mat  = crsMat.mat();
      PARCEL_INT *Rowptr = crsMat.rowptr_data();
      PARCEL_INT *crsCol = crsMat.crsCol_data();

      // 前処理行列の計算精度でデータを持ち最後に前処理行列LUに変換するためTprecon_calc_mat精度
      parcel::crsMat<Tprecon_calc_mat> LUmatrix_crsMat; 
      LUmatrix_crsMat.resize(N,NNZ,q_);
      vector<Tmat> &LUmatrix_mat = LUmatrix_crsMat.mat();
      vector_util vec_util;
      vec_util.convert<Tmat,Tprecon_calc_mat>
	(mat,LUmatrix_mat); // matrix --> precon_matrix
	
      LUmatrix_crsMat.rowptr_setValue(Rowptr);
      LUmatrix_crsMat.crsCol_setValue(crsCol);
      PARCEL_INT *LUmatrix_Rowptr = LUmatrix_crsMat.rowptr_data();
      PARCEL_INT *LUmatrix_crsCol = LUmatrix_crsMat.crsCol_data();

      const size_t work_group_size = 1; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数

      size_t *nhead = blockjacobi_buffer_.nhead_data();
      size_t *ntail = blockjacobi_buffer_.ntail_data();
      size_t *nnzL_ = blockjacobi_buffer_.nnzL_data();
      size_t *nnzU_ = blockjacobi_buffer_.nnzU_data();

      matrix<Tprecon_mat> &Lmatrix_matrix = blockjacobi_buffer_.Lmatrix();
      parcel::crsMat<Tprecon_mat> &Lmatrix_crsMat = *Lmatrix_matrix.crsMat_data();
      vector<Tprecon_mat> &Lmatrix_mat    = Lmatrix_crsMat.mat();
      PARCEL_INT   *Lmatrix_Rowptr = Lmatrix_crsMat.rowptr_data();
      PARCEL_INT   *Lmatrix_crsCol = Lmatrix_crsMat.crsCol_data();

      matrix<Tprecon_mat> &Umatrix_matrix = blockjacobi_buffer_.Umatrix();
      parcel::crsMat<Tprecon_mat> &Umatrix_crsMat = *Umatrix_matrix.crsMat_data();
      vector<Tprecon_mat> &Umatrix_mat    = Umatrix_crsMat.mat();
      PARCEL_INT   *Umatrix_Rowptr = Umatrix_crsMat.rowptr_data();
      PARCEL_INT   *Umatrix_crsCol = Umatrix_crsMat.crsCol_data();

      // --------------------------------------------------
      util_sycl::vector<PARCEL_INT> DiagIndex;
      DiagIndex.resize(N,q_);
      PARCEL_INT *DiagIndex_ptr = DiagIndex.data();
      q_[0].wait();
#if _A64FX_HIPSYCL_

      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
			   size_t ihead = nhead[group_id];
			   size_t itail = ntail[group_id];

			   // 対角要素のポインタ位置計算
			   size_t Nblock = itail-ihead;
			   for(size_t i = ihead;i<itail;i++){
			     for(PARCEL_INT j = Rowptr[i] + local_id; j < Rowptr[i+1]; j=j+work_group_size){
			       PARCEL_INT Col = crsCol[j];
			       if(i==Col){
				 DiagIndex_ptr[i] = j;
			       }
			     }
			   }
			 });
	});
#else
      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  size_t global_id = ii;
	  size_t ihead = nhead[global_id];
	  size_t itail = ntail[global_id];
	  
	  // 対角要素のポインタ位置計算
	  size_t Nblock = itail-ihead;
	  for(size_t i = ihead;i<itail;i++){
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      PARCEL_INT Col = crsCol[j];
	      if(i==Col){
		DiagIndex_ptr[i] = j;
	      }
	    }
	  }
    });
#endif
      q_[0].wait();

      // 元の行列と同じ精度でデータを持ち最後に前処理行列に変換する
      const auto Tmat_name             = typeid(Tmat).name();
      const auto Tprecon_mat_name      = typeid(Tprecon_mat).name();
      const auto Tprecon_calc_mat_name = typeid(Tprecon_calc_mat).name();

      auto *LU = LUmatrix_mat.data(); 
      double *LU_hi = (double *)LUmatrix_mat.data_hi(); 
      double *LU_lo = (double *)LUmatrix_mat.data_lo(); 

      bool ddflag_precon_calc = false;
      if( ddtype_name == Tprecon_calc_mat_name ){ 
	// 前処理行列の計算精度 :: dd 
	ddflag_precon_calc = true;
      }else{
	// 前処理行列の計算精度 :: d
	ddflag_precon_calc = false;
      }

      // 行列スケーリング
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      if( SCALING_MATRIX == true ){ 
	// std::cout<<"Matrix scaling"<<std::endl;
	scalingInv_vec_.resize(N,q_);
	auto *AmaxInv = scalingInv_vec_.data(); 
	crs_maxInv<
	  typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type,
	  double
	  >(N,LU,AmaxInv,LUmatrix_Rowptr,LUmatrix_crsCol);
      }

      // ILU分解開始
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
#else
      //      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  size_t global_id = ii;
	  size_t local_id  = 0;
	  size_t group_id  = ii;
#endif
			   size_t ihead = nhead[group_id];
			   size_t itail = ntail[group_id];
			   
			   if(local_id==0){
			     for(size_t i = ihead+1;i<itail;i++){
			       for(PARCEL_INT kptr = Rowptr[i];kptr<Rowptr[i+1];kptr++){
				 PARCEL_INT k = crsCol[kptr];
				 if( k>=i ){	
				   // kがiよりも大きくなったらループを抜ける
				   break;
				 }
				 if( k<ihead ){
				   // 計算領域外なら次のループ
				   continue;
				 }

				 PARCEL_INT kkptr = DiagIndex_ptr[k];
				 PARCEL_INT ikptr = kptr;
				 //dd対応
				 // if( ddtype_name == Tprecon_calc_mat_name ){ 
				 if( ddflag_precon_calc == true ){ 
#if 1
				   // LU[ikptr] = LU[ikptr] / LU[kkptr];
				   double dddiv_var1;
				   double LU_ikptr_hi = LU_hi[ikptr];
				   double LU_ikptr_lo = LU_lo[ikptr];
				   double LU_kkptr_hi = LU_hi[kkptr];
				   double LU_kkptr_lo = LU_lo[kkptr];
				   dddiv_calc1(LU_ikptr_hi,LU_ikptr_lo,LU_kkptr_hi,LU_kkptr_lo,LU_ikptr_hi,LU_ikptr_lo);
				   LU_hi[ikptr] = LU_ikptr_hi;
				   LU_lo[ikptr] = LU_ikptr_lo;
#else
				   LU[ikptr] = LU[ikptr] / LU[kkptr];
#endif
				 }else{
				   LU[ikptr] = LU[ikptr] / LU[kkptr];
				 }
				 
				 for(PARCEL_INT jptr=Rowptr[i];jptr<Rowptr[i+1];jptr++){
				   // i行j列を探索
				   PARCEL_INT j = crsCol[jptr];
				   if( itail<=j ){
				     // 計算領域を外れた場合は次のループ
				     break;
				   }
				   if( j<=k ){
				     // kよりも小さい
				     continue;
				   }

				   PARCEL_INT ijptr = jptr;
				   for(PARCEL_INT kjptr=Rowptr[k];kjptr<Rowptr[k+1];kjptr++){
				     PARCEL_INT kjCol = crsCol[kjptr];
				     if(kjCol==j){
				       // a(i,j) = a(i,j) -a(i,k)*a(k,j)
				       // dd対応
				       if( ddflag_precon_calc == true ){ 
					 // LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
#if 1

					 double ddmul_var1;
					 double ddsub_var1;
					 double LU_ijptr_hi = LU_hi[ijptr];
					 double LU_ijptr_lo = LU_lo[ijptr];
					 double LU_ikptr_hi = LU_hi[ikptr];
					 double LU_ikptr_lo = LU_lo[ikptr];
					 double LU_kjptr_hi = LU_hi[kjptr];
					 double LU_kjptr_lo = LU_lo[kjptr];
					 double c_hi = 0.0;
					 double c_lo = 0.0;
					 ddmul_calc1(LU_ikptr_hi,LU_ikptr_lo,LU_kjptr_hi,LU_kjptr_lo,c_hi,c_lo);
					 ddsub_calc1(LU_ijptr_hi,LU_ijptr_lo,c_hi,c_lo,LU_ijptr_hi,LU_ijptr_lo);
					 LU_hi[ijptr] = LU_ijptr_hi;
					 LU_lo[ijptr] = LU_ijptr_lo;
#else
					 LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
#endif
				       }else{
					 LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
				       }
				       //				       }else{
				       //					 LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
				       //				       }
				       
				       break;
				     }
				   }
				 }

			       }
			     }
			   }
			 });

#if _A64FX_HIPSYCL_
	});
#else

#endif
      q_[0].wait();

      // LUmatrixからLmatrix,Umatrixを作成する.
      // Lmatrixの対角項は1.0

      auto *L = Lmatrix_mat.data(); 
      auto *L_hi = Lmatrix_mat.data_hi(); 
      auto *L_lo = Lmatrix_mat.data_lo(); 
      auto *U = Umatrix_mat.data(); 
      auto *U_hi = Umatrix_mat.data_hi(); 
      auto *U_lo = Umatrix_mat.data_lo(); 

      bool ddflag_precon = false;
      if( ddtype_name == Tprecon_mat_name ){ 
	// 前処理行列の精度 :: dd 
 	ddflag_precon = true;
      }else{
	// 前処理行列の精度 :: d
	ddflag_precon = false;
      }
      
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
#else
      //      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  size_t global_id = ii;
	  size_t local_id  = 0;
	  size_t group_id  = ii;
#endif
			   size_t ihead = nhead[group_id];
			   size_t itail = ntail[group_id];
			   size_t nnzL  = nnzL_[group_id];
			   size_t nnzU  = nnzU_[group_id];

			   if(local_id==0){
			     
			     if(ihead==0){
			       Lmatrix_Rowptr[0] = 0;
			       Umatrix_Rowptr[0] = 0;
			     }

			     // 前処理行列と元の行列の精度により変換する必要がある
			     for(size_t i = ihead;i<itail;i++){

			       for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
				 PARCEL_INT Col = crsCol[j];
				 if( (Col<i) && (ihead<=Col) ){
				   if( ddflag_precon_calc == true ){ 
				     if( ddflag_precon == true ){ 
				       L_hi[nnzL] = LU_hi[j];
				       L_lo[nnzL] = LU_lo[j];
				     }else{
				       L[nnzL] = LU_hi[j] + LU_lo[j];
				     }
				   }else{
				     if( ddflag_precon == true ){ 
				       L_hi[nnzL] = LU[j];
				       L_lo[nnzL] = 0.0;
				     }else{
				       L[nnzL] = LU[j];
				     }
				   }
				   Lmatrix_crsCol[nnzL] = Col;	
				   nnzL = nnzL+1;
				   continue ;
				 }

				 if( i==Col ){
				   if( ddflag_precon_calc == true ){ 
				     if( ddflag_precon == true ){ 
				       L_hi[nnzL] = 1.0;
				       L_lo[nnzL] = 0.0;
				       U_hi[nnzU] = LU_hi[j];
				       U_lo[nnzU] = LU_lo[j];
				     }else{
				       L[nnzL] = 1.0;
				       U[nnzU] = LU_hi[j] + LU_lo[j] ;
				     }
				   }else{
				     if( ddflag_precon == true ){ 
				       L_hi[nnzL] = 1.0;
				       L_lo[nnzL] = 0.0;
				       U_hi[nnzU] = LU[j];
				       U_lo[nnzU] = 0.0;
				     }else{
				       L[nnzL] = 1.0;
				       U[nnzU] = LU[j];
				     }
				   }

				   Lmatrix_crsCol[nnzL] = Col;	
				   nnzL = nnzL+1;

				   Umatrix_crsCol[nnzU] = Col;	
				   nnzU = nnzU+1;
				 }
				 if( (Col>i) && (Col<itail) ){
				   if( ddflag_precon_calc == true ){ 
				     if( ddflag_precon == true ){ 
				       U_hi[nnzU] = LU_hi[j];
				       U_lo[nnzU] = LU_lo[j];
				     }else{
				       U[nnzU] = LU_hi[j] + LU_lo[j];
				     }
				   }else{
				     if( ddflag_precon == true ){ 
				       U_hi[nnzU] = LU[j];
				       U_lo[nnzU] = 0.0;
				     }else{
				       U[nnzU] = LU[j];
				     }
				   }
				   Umatrix_crsCol[nnzU] = Col;
				   nnzU = nnzU+1;
				   continue ;
				 }				   

			       }

			       Lmatrix_Rowptr[i+1] = nnzL;
			       Umatrix_Rowptr[i+1] = nnzU;
			     }
			   }

			 });
#if _A64FX_HIPSYCL_
	});
#else

#endif
      q_[0].wait();

#if 0
      std::vector<
	typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type
	> ILU_A(NNZ,-1);
      std::vector<PARCEL_INT> ILU_Rowptr(N+1,-1);
      std::vector<PARCEL_INT> ILU_crsCol(NNZ,-1);
      LUmatrix_crsMat.mat_getValue(ILU_A.data());

//      vec_util.convert<Tprecon_calc_mat,parcel_double>(ILU_A.data(),ILU_A); 

      LUmatrix_crsMat.rowptr_getValue(ILU_Rowptr.data());
      LUmatrix_crsMat.crsCol_getValue(ILU_crsCol.data());

      for(size_t i = 0;i<N;i++){
	for(size_t j = ILU_Rowptr[i];j<ILU_Rowptr[i+1];j++){
	  size_t col = ILU_crsCol[j];
	  std::cout<<i<<" , "<<col<<" : "<<ILU_A[j]<<std::endl;
	}
	std::cout<<" --------- "<<std::endl;
      }
#endif

#if 0
      size_t nnzL_size = Lmatrix_mat.size();
      // std::vector<double> L_A(nnzL_size,-1);
      std::vector<
	typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
	> L_A(nnzL_size,-1);
      std::vector<PARCEL_INT> L_Rowptr(N+1,-1);
      std::vector<PARCEL_INT> L_crsCol(nnzL_size,-1);
      Lmatrix_crsMat.mat_getValue(L_A.data()); 
      Lmatrix_crsMat.rowptr_getValue(L_Rowptr.data()); 
      Lmatrix_crsMat.crsCol_getValue(L_crsCol.data()); 

      size_t nnzU_size = Umatrix_mat.size();
      //      std::vector<double> U_A(nnzU_size,-1);
      std::vector<
	typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
	> U_A(nnzU_size,-1);
      std::vector<PARCEL_INT> U_Rowptr(N+1,-1);
      std::vector<PARCEL_INT> U_crsCol(nnzU_size,-1);
      Umatrix_crsMat.mat_getValue(U_A.data()); 
      Umatrix_crsMat.rowptr_getValue(U_Rowptr.data()); 
      Umatrix_crsMat.crsCol_getValue(U_crsCol.data()); 

      //      std::cout<<"nnzL_size = "<<nnzL_size<<std::endl;
      //      std::cout<<"nnzU_size = "<<nnzU_size<<std::endl;
      for(size_t i = 0;i<N;i++){
	// std::cout<<"L "<<i<<" : "<<L_Rowptr[i]<<" --> "<<L_Rowptr[i+1]<<std::endl;
	std::cout<<"L "<<i<<" : "<<L_Rowptr[i]<<" --> "<<L_Rowptr[i+1]<<std::endl;
	for(size_t j = L_Rowptr[i];j<L_Rowptr[i+1];j++){
	  size_t col = L_crsCol[j];
	  std::cout<<i<<" , "<<col<<" : "<<L_A[j]<<std::endl;
	}
	std::cout<<" --------- "<<std::endl;
      }

      for(size_t i = 0;i<N;i++){
	//	std::cout<<"U "<<i<<" : "<<U_Rowptr[i]<<" --> "<<U_Rowptr[i+1]<<std::endl;
	std::cout<<"U "<<i<<" : "<<U_Rowptr[i]<<" --> "<<U_Rowptr[i+1]<<std::endl;
	for(size_t j = U_Rowptr[i];j<U_Rowptr[i+1];j++){
	  size_t col = U_crsCol[j];
	  std::cout<<i<<" , "<<col<<" : "<<U_A[j]<<std::endl;
	}
	std::cout<<" --------- "<<std::endl;
      }

#endif

      return ;

    }

    void get_blockjacobi_LUmatrix(
				  parcel::diaMat<Tmat> &diaMat
				  ){

      sycl::queue *q_ = diaMat.queue();
      setQueue(q_);
      size_t N   = diaMat.n();
      size_t NND = diaMat.nnd();

      vector<Tmat> &mat  = diaMat.mat();
      PARCEL_INT *offset = diaMat.offset_data();

      util_sycl::vector<PARCEL_INT> &offset_vec = diaMat.offset();
      std::vector<PARCEL_INT> offset_cpu(NND,0);
      offset_vec.getValue( offset_cpu.data() );

      // 前処理行列の計算精度でデータを持ち最後に前処理行列LUに変換するためTprecon_calc_mat精度
      parcel::diaMat<Tprecon_calc_mat> LUmatrix_diaMat; 
      LUmatrix_diaMat.resize(N,NND,q_);

      vector<Tmat> &LUmatrix_mat = LUmatrix_diaMat.mat();
      vector_util vec_util;
      vec_util.convert<Tmat,Tprecon_calc_mat>
	(mat,LUmatrix_mat); // matrix --> precon_matrix
      LUmatrix_diaMat.offset_setValue( offset_cpu.data() );

      const size_t work_group_size = 1; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数

      size_t *nhead = blockjacobi_buffer_.nhead_data();
      size_t *ntail = blockjacobi_buffer_.ntail_data();

#if 0
      std::vector<size_t> nhead_cpu(BLOCK_JACOBI_NUMBLOCK_,-1);
      std::vector<size_t> ntail_cpu(BLOCK_JACOBI_NUMBLOCK_,-1);
      nhead_cpu[0]=10;
      ntail_cpu[0]=11;
      std::cout<<" @ 1 @ : head,tail = "<<nhead_cpu[0]<<" --> "<<ntail_cpu[0]<<std::endl;
      //      blockjacobi_buffer_.nhead_getValue(nhead_cpu.data());
      //      blockjacobi_buffer_.ntail_getValue(ntail_cpu.data());
      util_sycl::vector<size_t> &nhead_data = blockjacobi_buffer_.nhead();
      util_sycl::vector<size_t> &ntail_data = blockjacobi_buffer_.ntail();
      nhead_data.getValue(nhead_cpu.data());
      ntail_data.getValue(ntail_cpu.data());
      std::cout<<" @ 2 @ : head,tail = "<<nhead_cpu[0]<<" --> "<<ntail_cpu[0]<<std::endl;
#endif

      matrix<Tprecon_mat> &Lmatrix_matrix = blockjacobi_buffer_.Lmatrix();
      parcel::diaMat<Tprecon_mat> &Lmatrix_diaMat = *Lmatrix_matrix.diaMat_data();
      vector<Tprecon_mat> &Lmatrix_mat            = Lmatrix_diaMat.mat();
      PARCEL_INT *Lmatrix_offset = Lmatrix_diaMat.offset_data();
      PARCEL_INT nndL = Lmatrix_diaMat.nnd();
      matrix<Tprecon_mat> &Umatrix_matrix = blockjacobi_buffer_.Umatrix();
      parcel::diaMat<Tprecon_mat> &Umatrix_diaMat = *Umatrix_matrix.diaMat_data();
      vector<Tprecon_mat> &Umatrix_mat            = Umatrix_diaMat.mat();
      PARCEL_INT *Umatrix_offset = Umatrix_diaMat.offset_data();


      PARCEL_INT idiag_ptr_ = -1; 
      for(PARCEL_INT j = 0;j<NND;j++){
	PARCEL_INT offset_ = offset_cpu[j];
	if(offset_==0){
	  idiag_ptr_ = j;
	}
      }
      const PARCEL_INT idiag_ptr = idiag_ptr_;

      // 元の行列と同じ精度でデータを持ち最後に前処理行列に変換する
      const auto Tmat_name             = typeid(Tmat).name();
      const auto Tprecon_mat_name      = typeid(Tprecon_mat).name();
      const auto Tprecon_calc_mat_name = typeid(Tprecon_calc_mat).name();

      auto *LU    = LUmatrix_mat.data(); 
      auto *LU_hi = LUmatrix_mat.data_hi(); 
      auto *LU_lo = LUmatrix_mat.data_lo(); 

      // ブロック領域以外をゼロにする

      bool ddflag_precon_calc = false;
      if( ddtype_name == Tprecon_calc_mat_name ){ 
	// 前処理行列の計算精度 :: dd 
	ddflag_precon_calc = true;
      }else{
	// 前処理行列の計算精度 :: d
	ddflag_precon_calc = false;
      }
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   const size_t global_id = item.get_global_id(0);
			   const size_t local_id  = item.get_local_id(0);
			   const size_t group_id  = item.get_group(0);
#else
      const size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  const size_t global_id = ii;
	  const size_t local_id  = 0;
	  const size_t group_id  = ii;
#endif
	  if(local_id==0){
	  const PARCEL_INT ihead = nhead[group_id];
	  const PARCEL_INT itail = ntail[group_id];	  

	  for(PARCEL_INT j = 0;j<NND;j++){
	  PARCEL_INT offset_ = offset[j];
	  for(PARCEL_INT i = ihead;i<itail;i++){
	    PARCEL_INT ii = i + offset_;
	    if( (ii<ihead) || (itail<=ii) ){
	      if( ddflag_precon_calc == true ){ 
		LU_hi[i + j*N] = 0.0;
		LU_lo[i + j*N] = 0.0;
	      }else{
		LU[i + j*N] = 0.0;
	      }
	    }
	  }
	  }
	  }

	});
#if _A64FX_HIPSYCL_
	});
#else

#endif

      // std::cout<<" ddflag_precon_calc = "<<ddflag_precon_calc<<" , true = "<<true<<std::endl;
      // 行列スケーリング
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      if( SCALING_MATRIX == true ){ 
	// std::cout<<"Matrix scaling"<<std::endl;
	scalingInv_vec_.resize(N,q_);
	auto *AmaxInv = scalingInv_vec_.data(); 
	dia_maxInv<
	  typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type,
	  double
	  >(N,NND,LU,AmaxInv);
      }
      q_[0].wait();

      // ILU分解開始
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   const size_t global_id = item.get_global_id(0);
			   const size_t local_id  = item.get_local_id(0);
			   const size_t group_id  = item.get_group(0);
#else
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  const size_t global_id = ii;
	  const size_t local_id  = 0;
	  const size_t group_id  = ii;
#endif

	  if(local_id==0){

	  const PARCEL_INT ihead = nhead[group_id];
	  const PARCEL_INT itail = ntail[group_id];	  
	  //	  printf(" group_id %d : ihead,itail = %d,%d \n",group_id,nhead[group_id],ntail[group_id]);
	    for(PARCEL_INT i = ihead+1;i<itail;i++){
	      for(PARCEL_INT kptr = 0;kptr<NND;kptr++){
		PARCEL_INT k = i + offset[kptr];

		if( k>=i ){	
		  // kがiよりも大きくなったらループを抜ける
		  break;
		}

		// if( k<ihead ){
		if( (k<ihead) || (itail<=k) ){
		  // 計算領域外なら次のループ
		  continue;
		}

		// PARCEL_INT kkptr = DiagIndex_ptr[k];
		PARCEL_INT ikptr = i + kptr*N;
		PARCEL_INT kkptr = k + idiag_ptr*N;
		// 

		//dd対応
		// if( ddtype_name == Tprecon_calc_mat_name ){ 
		if( ddflag_precon_calc == true ){ 
		  // LU[ikptr] = LU[ikptr] / LU[kkptr];
		  double dddiv_var1;
		  double LU_ikptr_hi = LU_hi[ikptr];
		  double LU_ikptr_lo = LU_lo[ikptr];
		  double LU_kkptr_hi = LU_hi[kkptr];
		  double LU_kkptr_lo = LU_lo[kkptr];
		  // -------
		  
		  dddiv_calc1(LU_ikptr_hi,LU_ikptr_lo,LU_kkptr_hi,LU_kkptr_lo,LU_ikptr_hi,LU_ikptr_lo);
		  LU_hi[ikptr] = LU_ikptr_hi;
		  LU_lo[ikptr] = LU_ikptr_lo;
		  //		  LU[ikptr] = LU[ikptr] / LU[kkptr];
		}else{
		   LU[ikptr] = LU[ikptr] / LU[kkptr];
		}

		// printf(" i,k = %d,%d \n",i,k);
		for(PARCEL_INT jptr=0;jptr<NND;jptr++){
		  // i行j列を探索
		  PARCEL_INT j = i + offset[jptr];
		  // if( itail<=j ){
		  if( (j<ihead) || (itail<=j) ){
		    // 計算領域を外れた場合は次のループ
		    // break;
		    continue;
		  }
		  if( j<=k ){
		    // kよりも小さい
		    continue;
		  }

		  PARCEL_INT ijptr = i + jptr*N;
		  for(int lptr=0;lptr<NND;lptr++){
		    PARCEL_INT kjCol = k + offset[lptr];
		    PARCEL_INT kjptr = k + lptr*N;
		    if(kjCol==j){
		      // a(i,j) = a(i,j) -a(i,k)*a(k,j)
		      // dd対応
		      if( ddflag_precon_calc == true ){ 
			// LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
			double ddmul_var1;
			double ddsub_var1;
			double LU_ijptr_hi = LU_hi[ijptr];
			double LU_ijptr_lo = LU_lo[ijptr];
			double LU_ikptr_hi = LU_hi[ikptr];
			double LU_ikptr_lo = LU_lo[ikptr];
			double LU_kjptr_hi = LU_hi[kjptr];
			double LU_kjptr_lo = LU_lo[kjptr];
			double c_hi = 0.0;
			double c_lo = 0.0;
			ddmul_calc1(LU_ikptr_hi,LU_ikptr_lo,LU_kjptr_hi,LU_kjptr_lo,c_hi,c_lo);
			ddsub_calc1(LU_ijptr_hi,LU_ijptr_lo,c_hi,c_lo,LU_ijptr_hi,LU_ijptr_lo);
			LU_hi[ijptr] = LU_ijptr_hi;
			LU_lo[ijptr] = LU_ijptr_lo;
		      }else{
			LU[ijptr] = LU[ijptr] - LU[ikptr] * LU[kjptr];
		      }
		      break;
		    }
		  }
		}

	      }
	    }

	  }

	});

#if _A64FX_HIPSYCL_
	});
#else

#endif

      q_[0].wait();

#if 0
	  const size_t LU_mat_size = N*NND;
	  std::vector<
	    typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type
	    > LU_cpu(LU_mat_size,0.0); 
	  LUmatrix_mat.getValue( LU_cpu.data() );
	q_[0].wait();
	  for(PARCEL_INT i = 0;i<N;i++){
	    for(PARCEL_INT j = 0;j<NND;j++){
	      PARCEL_INT offset_ = offset[j];
	      //	      if( ( -1<(i+offset_) ) && ( (i+offset_) < N ) )
	      std::cout<<"i,j : "<<i<<","<<j<<" = "<<LU_cpu[i+j*N]<<std::endl;
	      //	      std::cout<<"i,j : "<<i<<","<<j<<" = "<<i+j*N<<" : size = "<<LU_cpu.size()<<std::endl;
	    }
	    std::cout<<"-------------"<<std::endl;
	  }
#endif
      q_[0].wait();

      // L,Uに分割
      auto *L = Lmatrix_mat.data(); 
      auto *L_hi = Lmatrix_mat.data_hi(); 
      auto *L_lo = Lmatrix_mat.data_lo(); 
      auto *U = Umatrix_mat.data(); 
      auto *U_hi = Umatrix_mat.data_hi(); 
      auto *U_lo = Umatrix_mat.data_lo(); 

      bool ddflag_precon = false;
      if( ddtype_name == Tprecon_mat_name ){ 
	// 前処理行列の精度 :: dd 
 	ddflag_precon = true;
      }else{
	// 前処理行列の精度 :: d
	ddflag_precon = false;
      }

#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   const size_t global_id = item.get_global_id(0);
			   const size_t local_id  = item.get_local_id(0);
			   const size_t group_id  = item.get_group(0);
#else
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  const size_t global_id = ii;
	  const size_t local_id  = 0;
	  const size_t group_id  = ii;
#endif

	  if(local_id==0){
	    PARCEL_INT nndU_ = 1; //対角は必ずnndU=0なので1始まり
	    PARCEL_INT nndL_ = 0;
	    for(PARCEL_INT j = 0;j<NND;j++){
	      PARCEL_INT offset_ = offset[j];
	      if(offset_<0){
		if( ddflag_precon_calc == true ){ 
		  if( ddflag_precon == true ){ 
 		    for(PARCEL_INT i = 0;i<N;i++){
		      L_hi[i + nndL_*N] = LU_hi[i + j*N];
		      L_lo[i + nndL_*N] = LU_lo[i + j*N];
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      L[i + nndL_*N] = LU_hi[i + j*N] + LU_lo[i + j*N];
		    }
		  }
		}else{
		  if( ddflag_precon == true ){ 
		    for(PARCEL_INT i = 0;i<N;i++){
		      L_hi[i + nndL_*N] = LU[i + j*N];
		      L_lo[i + nndL_*N] = 0.0;
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      L[i + nndL_*N] = LU[i + j*N];
		    }
		  }
		}
		nndL_ = nndL_ + 1;
		continue;
	      }

	      if(0==offset_){
		if( ddflag_precon_calc == true ){ 
		  if( ddflag_precon == true ){ 
 		    for(PARCEL_INT i = 0;i<N;i++){
		      L_hi[i + (nndL-1)*N] = 1.0;
		      L_lo[i + (nndL-1)*N] = 0.0;
		      U_hi[i + 0*N]        = LU_hi[i + j*N];
		      U_lo[i + 0*N]        = LU_lo[i + j*N];
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      L[i + (nndL-1)*N] = 1.0;
		      U[i + 0*N]        = LU_hi[i + j*N] + LU_lo[i + j*N];
		    }
		  }
		}else{
		  if( ddflag_precon == true ){ 
		    for(PARCEL_INT i = 0;i<N;i++){
		      L_hi[i + (nndL-1)*N] = 1.0;
		      L_lo[i + (nndL-1)*N] = 0.0;
		      U_hi[i + 0*N]        = LU[i + j*N];
		      U_lo[i + 0*N]        = 0.0;
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      L[i + (nndL-1)*N] = 1.0;
		      U[i + 0*N]        = LU[i + j*N];
		    }
		  }
		}
		continue;
	      }

	      if(0<offset_){
		if( ddflag_precon_calc == true ){ 
		  if( ddflag_precon == true ){ 
 		    for(PARCEL_INT i = 0;i<N;i++){
		      U_hi[i + nndU_*N] = LU_hi[i + j*N];
		      U_lo[i + nndU_*N] = LU_lo[i + j*N];
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      U[i + nndU_*N] = LU_hi[i + j*N] + LU_lo[i + j*N];
		    }
		  }
		}else{
		  if( ddflag_precon == true ){ 
		    for(PARCEL_INT i = 0;i<N;i++){
		      U_hi[i + nndU_*N] = LU[i + j*N];
		      U_lo[i + nndU_*N] = 0.0;
		    }
		  }else{
		    for(PARCEL_INT i = 0;i<N;i++){
		      U[i + nndU_*N] = LU[i + j*N];
		    }
		  }
		}
		nndU_ = nndU_ + 1;
		continue;
	      }
	    }
	  }

	});
#if _A64FX_HIPSYCL_
	});
#else

#endif

      q_[0].wait();

#if 0
      const size_t L_mat_size = N*NND;
      const size_t nndL_size = blockjacobi_buffer_.get_nndL_size();
      std::vector<
	typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
	> L_cpu(L_mat_size,0.0); 
      Lmatrix_mat.getValue( L_cpu.data() );
      q_[0].wait();

      for(PARCEL_INT i = 0;i<N;i++){
	  for(PARCEL_INT j = 0;j<nndL_size;j++){
	  PARCEL_INT offset_ = offset[j];
	  std::cout<<"L : i,j : "<<i<<","<<j<<" = "<<L_cpu[i+j*N]<<std::endl;
	}
	  std::cout<<"-------------"<<std::endl;
	}
#endif

#if 0
      const size_t U_mat_size = N*NND;
      const size_t nndU_size = blockjacobi_buffer_.get_nndU_size();
      std::vector<
	typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
	> U_cpu(U_mat_size,0.0); 
      Umatrix_mat.getValue( U_cpu.data() );
      q_[0].wait();

      for(PARCEL_INT i = 0;i<N;i++){
	  for(PARCEL_INT j = 0;j<nndU_size;j++){
	  PARCEL_INT offset_ = offset[j];
	  std::cout<<"U : i,j : "<<i<<","<<j<<" = "<<U_cpu[i+j*N]<<std::endl;
	}
	  std::cout<<"-------------"<<std::endl;
	}
#endif

      return ;

    }

#include "parcel_precon_ddblockjacobi.hpp"

    // CRS 
    template<class T_matrix,class T_vec,class T_work,class T_calc>
    void blockjacobi(
		     sycl::queue *q_,
		     PARCEL_INT N,
		     T_vec  *r,
		     T_work *y,
		     T_vec  *z,
		     T_matrix *L,
		     PARCEL_INT *L_Rowptr,
		     PARCEL_INT *L_crsCol,
		     T_matrix *U,
		     PARCEL_INT *U_Rowptr,
		     PARCEL_INT *U_crsCol,
		     double *scaling_Inv
		     ){

#if _A64FX_HIPSYCL_
      const size_t nblock      = 12;
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif

      const bool SCALING_MATRIX = SCALING_MATRIX_;
      const bool SCALING_VECTOR = SCALING_VECTOR_;

      const size_t work_group_size = 1; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
      size_t *nhead = blockjacobi_buffer_.nhead_data();
      size_t *ntail = blockjacobi_buffer_.ntail_data();
      // for(PARCEL_INT i=0;i<BLOCK_JACOBI_NUMBLOCK_;i++){
      // 	std::cout<<i<<" @ head,tail : "<<nhead[i]<<","<<ntail[i]<<std::endl;
      // }
      double rnorm_ = 1.0;

      if( SCALING_VECTOR == true ){ 
	if( SCALING_MATRIX == true ){ 
	  rnorm_ = max<T_vec,double,double>(N,r,N,scaling_Inv,N);
	}else{
	  rnorm_ = max<T_vec,double>(N,r,N);
	}
      }
      
      const double rnorm = rnorm_;
      q_[0].wait();
      //      return ;
      //        std::cout<<" rnorm = "<<rnorm<<std::endl;
      //	return ;
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
#else

      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  size_t global_id = ii;
	  size_t local_id  = 0;
	  size_t group_id  = ii;
#endif
			   PARCEL_INT ihead = nhead[group_id];
			   PARCEL_INT itail = ntail[group_id];

			   if(local_id==0){
#if 0
			     for(size_t i = ihead;i<itail;i++){
			       z[i] = r[i];
			     }
#else
			     // 1. solve Ly = r
			     for(PARCEL_INT i = ihead;i<itail;i++){
			       // Lの対角要素は1.0が条件
			       T_calc y_ = r[i] / rnorm;
			       if( SCALING_MATRIX == true ){ 
				 y_ = y_*scaling_Inv[i];
			       }
			       for(PARCEL_INT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){
				 PARCEL_INT Col = L_crsCol[j];
				 if(Col!=i){
				   T_calc yL_ = y[Col];
				   T_calc L_  = L[j];
				   y_ = y_ - yL_*L_;
				   //printf(" %d :    L_ = %f, y_ = %f \n",i,y_,L_);
				 }
			       }
			       // z[i] = y_;
			       y[i] = y_;
			       //printf("r[%d] = %f, y[%d] = %f\n",i,r[i],i,y[i]);
			     }

			     // 2. solve Uz = y
			     // for(int i = ntail-1;i>(nhead-1);i--){
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){
			       T_calc z_ = y[i];
			       T_calc DiagInv = 1.0;
			       for(PARCEL_INT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){
				 PARCEL_INT Col = U_crsCol[j];
				 if(Col==i){
				   DiagInv = 1.0/U[j];
				 }else{
				   T_calc zL_ = y[Col];
				   T_calc U_  = U[j];
				   z_ = z_ - zL_*U_;
				 }
				   
			       }
			       z_   = z_*DiagInv;
			       y[i] = z_;
			       z[i] = z_;
			       //printf("r[%d] = %f, z[%d] = %f\n",i,r[i],i,z[i]);
			     }


#endif
			   }

			 });
#if _A64FX_HIPSYCL_
	});
#else

#endif
      q_[0].wait();

      if( SCALING_VECTOR == true ){ 
	const size_t N_ = N ;
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
		q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z[i] = z[i] * rnorm;
		//		x[i] = x[i] * x[i] * x[i] ;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
	q_[0].wait();

      }

      return ;
    }

    // DIA 
    template<class T_matrix,class T_vec,class T_work,class T_calc>
    void blockjacobi(
		     sycl::queue *q_,
		     PARCEL_INT N,
		     T_vec  *r,
		     T_work *y,
		     T_vec  *z,
		     T_matrix *L,
		     PARCEL_INT *L_offset,
		     PARCEL_INT nndL,
		     T_matrix *U,
		     PARCEL_INT *U_offset,
		     PARCEL_INT nndU,
		     double *scaling_Inv
		     ){

#if _A64FX_HIPSYCL_
      const size_t nblock      = 12;
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif

      const auto Tmat_name  = typeid(T_matrix).name();
      const auto Tcalc_name = typeid(T_calc).name();

      const bool SCALING_MATRIX = SCALING_MATRIX_;
      const bool SCALING_VECTOR = SCALING_VECTOR_;

      const size_t work_group_size = 1; // simd lengh
      const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
      const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
      size_t *nhead = blockjacobi_buffer_.nhead_data();
      size_t *ntail = blockjacobi_buffer_.ntail_data();

      double rnorm_ = 1.0;
      if( SCALING_VECTOR == true ){ 
	setQueue(q_);
	if( SCALING_MATRIX == true ){ 
	  rnorm_ = max<T_vec,double,double>(N,r,N,scaling_Inv,N);
	}else{
	  rnorm_ = max<T_vec,double>(N,r,N);
	}
	q_[0].wait();
      }
      const double rnorm = rnorm_;
 
#if _A64FX_HIPSYCL_
      q_[0].submit([&] (sycl::handler &h){
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
			 [=](sycl::nd_item<1> item) {
			   size_t global_id = item.get_global_id(0);
			   size_t local_id  = item.get_local_id(0);
			   size_t group_id  = item.get_group(0);
#else
      size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
	  // work_group_size=1限定で動作
	  size_t global_id = ii;
	  size_t local_id  = 0;
	  size_t group_id  = ii;
#endif
	  PARCEL_INT ihead = nhead[group_id];
	  PARCEL_INT itail = ntail[group_id];
			   
	  // まずは並列化無し
	  if(local_id==0){
#if 0
	    for(size_t i = ihead;i<itail;i++){
	      z[i] = r[i];
	    }
#else

	    // 1. solve Ly = r
	    for(PARCEL_INT i = ihead;i<itail;i++){
	      // Lの対角要素は1.0が条件
	      T_calc y_ = r[i] / rnorm;
	      if( SCALING_MATRIX == true ){ 
		y_ = y_*scaling_Inv[i];
	      }
	      // 非対角ループ
	      for(PARCEL_INT j = 0; j < nndL-1; j++){
		const PARCEL_INT offset = L_offset[j];
		const PARCEL_INT Col    = i + offset;
		if( ihead<=Col ){
		  const T_calc yL_ = y[Col];
		  const T_calc L_  = L[i+j*N];
		  y_ = y_ - yL_*L_;
		  //printf(" %d :    L_ = %f, y_ = %f \n",i,y_,L_);
		  //  }else{
		  //printf(" %d : ihead = %d, Col = %d, \n",i,ihead,Col);
	        }
	      }
	      y[i] = y_;
	      //printf("r[%d] = %f, y[%d] = %f\n",i,r[i],i,y[i]);
	    }

	    // 2. solve Uz = y
	    // for(int i = ntail-1;i>(nhead-1);i--){
	    for(PARCEL_INT i = itail-1;i>=(ihead);i--){
	      T_calc z_ = y[i];
	      T_calc DiagInv = 1.0;
	      
	      // 非対角ループ
	      for(PARCEL_INT j = 1; j < nndU; j++){
		const PARCEL_INT offset = U_offset[j];
		const PARCEL_INT Col = i + offset;
		if( Col<itail ){
		  T_calc zL_ = y[Col];
		  T_calc U_  = U[i+j*N];
		  z_ = z_ - zL_*U_;
		}
	      }

	      // 対角要素の逆数
	      DiagInv = 1.0/U[i+0*N];
	      z_   = z_*DiagInv;
	      y[i] = z_;
	      z[i] = z_;
	      //printf("r[%d] = %f, z[%d] = %f\n",i,r[i],i,z[i]);
	    }
#endif
	  }
	  
	});

#if _A64FX_HIPSYCL_
	});
#else

#endif
      q_[0].wait();

      if( SCALING_VECTOR == true ){ 
	const size_t N_ = N ;
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
		q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z[i] = z[i] * rnorm;
		//		x[i] = x[i] * x[i] * x[i] ;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
	q_[0].wait();

      }

      return ;
    }

   void solver_main(
		    vector<Tvec> &b_,
		    vector<Tvec> &x_,
		    parcel::crsMat<Tprecon_mat> &Lmatrix_crsMat,
		    parcel::crsMat<Tprecon_mat> &Umatrix_crsMat
		    ){
#if 0
	  vector_util vec_util;
	  vec_util.convert<Tvec,Tvec>(b_,x_);	
	  //	  return ;
#endif

#if 0
     const size_t N = x_.size();
#else     
     //     const size_t N = scalingInv_vec_.size();
     const size_t N   = Lmatrix_crsMat.n();
#endif     
     vector<Tprecon_vec> &work = blockjacobi_buffer_.work();
	const auto Tcalc_name       = typeid(Tcalc).name();
	const auto Tvec_name        = typeid(Tvec).name();
	const auto Tprecon_mat_name = typeid(Tprecon_mat).name();
	const auto Tprecon_vec_name = typeid(Tprecon_vec).name();

	vector<Tprecon_mat> &Lmatrix_mat = Lmatrix_crsMat.mat();
	PARCEL_INT *Lmatrix_Rowptr = Lmatrix_crsMat.rowptr_data();
	PARCEL_INT *Lmatrix_crsCol = Lmatrix_crsMat.crsCol_data();
      
	vector<Tprecon_mat> &Umatrix_mat = Umatrix_crsMat.mat();
	PARCEL_INT *Umatrix_Rowptr = Umatrix_crsMat.rowptr_data();
	PARCEL_INT *Umatrix_crsCol = Umatrix_crsMat.crsCol_data();

	// LUz = r
	// 1. solve Ly = r
	// 2. solve Uz = y
	sycl::queue *q_ = Lmatrix_crsMat.queue();

	if( ddtype_name == Tcalc_name ){ 
	  // double - double 
	  //  std::cout<<" double-double BlockJacobi "<<std::endl;

	  auto *L = Lmatrix_mat.data(); 
	  auto *U = Umatrix_mat.data(); 
	  auto *z = x_.data();
	  auto *r = b_.data();
	  auto *y = work.data();
	  auto *scalingInv_vec = scalingInv_vec_.data();

	  double *L_hi = (double *) Lmatrix_mat.data_hi(); 
	  double *L_lo = (double *) Lmatrix_mat.data_lo(); 
	  double *U_hi = (double *) Umatrix_mat.data_hi(); 
	  double *U_lo = (double *) Umatrix_mat.data_lo(); 
	  double *z_hi = (double *) x_.data_hi();
	  double *z_lo = (double *) x_.data_lo();
	  double *r_hi = (double *) b_.data_hi();
	  double *r_lo = (double *) b_.data_lo();
	  double *y_hi = (double *) work.data_hi();
	  double *y_lo = (double *) work.data_lo();

	    // ddblockjacobi<
	    //   typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
	    //   typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
	    //   typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
	    //   >(){	    };

	  if( ddtype_name ==  Tprecon_mat_name){ 
	    if( ddtype_name == Tvec_name){ 
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : dd
		// Tvec : dd
		// Tprecon_vec_name : dd
		ddblockjacobi(
			      q_,
			      N,
			      r_hi,r_lo,
			      y_hi,y_lo,
			      z_hi,z_lo,
			      L_hi,L_lo,
			      Lmatrix_Rowptr,Lmatrix_crsCol,
			      U_hi,U_lo,
			      Umatrix_Rowptr,Umatrix_crsCol,	      
			      scalingInv_vec);

	      }else{
		// Tprecon_mat : dd
		// Tvec : dd
		// Tprecon_vec_name : d
 		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y,
		    z_hi,z_lo,
		    L_hi,L_lo,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U_hi,U_lo,
		    Umatrix_Rowptr,Umatrix_crsCol,
			      scalingInv_vec);
	      }
	    }else{
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : dd
		// Tvec : d
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y_hi,y_lo,
		    z,
		    L_hi,L_lo,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U_hi,U_lo,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);

	      }else{
		// Tprecon_mat : dd
		// Tvec : d
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y,
		    z,
		    L_hi,L_lo,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U_hi,U_lo,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);

	      }
	    }
	  }else{
	    if( ddtype_name == Tvec_name){ 
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : d
		// Tvec : dd
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y_hi,y_lo,
		    z_hi,z_lo,
		    L,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);

	      }else{
		// Tprecon_mat : d
		// Tvec : dd
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y,
		    z_hi,z_lo,
		    L,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);

	      }
	    }else{
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : d
		// Tvec : d
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y_hi,y_lo,
		    z,
		    L,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);
	      }else{
		//		std::cout<<" double all "<<std::endl;
		// Tprecon_mat : d
		// Tvec : d
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y,
		    z,
		    L,
		    Lmatrix_Rowptr,Lmatrix_crsCol,
		    U,
		    Umatrix_Rowptr,Umatrix_crsCol,
		    scalingInv_vec);
		return ;
	      }
	    }

	  }
#if 0
	  blockjacobi<
	    double,double,double,double
	    >(
	      q_,
	      r_hi,
	      y_hi,
	      z_hi,
	      L_hi,
	      Lmatrix_Rowptr,Lmatrix_crsCol,
	      U_hi,
	      Umatrix_Rowptr,Umatrix_crsCol	      
	      );
#endif
	}else{

	  auto *L = Lmatrix_mat.data(); 
	  auto *U = Umatrix_mat.data(); 
	  auto *z = x_.data();
	  auto *r = b_.data();
	  auto *y = work.data();
	  auto *scalingInv_vec = scalingInv_vec_.data();

	  // const auto Tvec_name              = typeid(T_vec).name();
	  // const auto Tprecon_mat_name       = typeid(T_mat).name();
	  // const auto Tprecon_work_vec_name  = typeid(T_work_vec).name();
	  // const auto Tprecon_calc_name      = typeid(T_calc).name();
	  
	  blockjacobi<
	    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tcalc) >::type
	    >(
	      q_,
	      N,
	      r,
	      y,
	      z,
	      L,
	      Lmatrix_Rowptr,
	      Lmatrix_crsCol,
	      U,
	      Umatrix_Rowptr,
	      Umatrix_crsCol,
	      scalingInv_vec
	      );
	}

     return ;
   }

   void solver_main(
		    vector<Tvec> &b_,
		    vector<Tvec> &x_,
		    parcel::diaMat<Tprecon_mat> &Lmatrix_diaMat,
		    parcel::diaMat<Tprecon_mat> &Umatrix_diaMat
		    ){
#if 0
     const size_t N = x_.size();
#else     
     //     const size_t N = scalingInv_vec_.size();
     const size_t N = Lmatrix_diaMat.n();
#endif     

     vector<Tprecon_vec> &work = blockjacobi_buffer_.work();
	const auto Tcalc_name       = typeid(Tcalc).name();
	const auto Tvec_name        = typeid(Tvec).name();
	const auto Tprecon_mat_name = typeid(Tprecon_mat).name();
	const auto Tprecon_vec_name = typeid(Tprecon_vec).name();

	vector<Tprecon_mat> &Lmatrix_mat = Lmatrix_diaMat.mat();
	PARCEL_INT *Lmatrix_offset = Lmatrix_diaMat.offset_data();      
	PARCEL_INT nndL = Lmatrix_diaMat.nnd();

	vector<Tprecon_mat> &Umatrix_mat = Umatrix_diaMat.mat();
	PARCEL_INT *Umatrix_offset = Umatrix_diaMat.offset_data();
	PARCEL_INT nndU = Umatrix_diaMat.nnd();

	// LUz = r
	// 1. solve Ly = r
	// 2. solve Uz = y
	sycl::queue *q_ = Lmatrix_diaMat.queue();

	auto *L = Lmatrix_mat.data(); 
	auto *U = Umatrix_mat.data(); 
	auto *z = x_.data();
	auto *r = b_.data();
	auto *y = work.data();
	auto *scalingInv_vec = scalingInv_vec_.data();
#if 0
#if 1
	ddblockjacobi<
	  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
	  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
	  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
	  >(
	    q_,
	    N,
	    r,
	    y,
	    z,
	    L,
	    Lmatrix_offset,
	    nndL,
	    U,
	    Umatrix_offset,
	    nndU,
	    scalingInv_vec
	    );

#else
	blockjacobi<
	  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
	  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
	  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type,
	  typename std::result_of<parcel::parcel_type_to_baseType(Tcalc) >::type
	  >(
	    q_,
	    N,
	    r,
	    y,
	    z,
	    L,
	    Lmatrix_offset,
	    nndL,
	    U,
	    Umatrix_offset,
	    nndU,
	    scalingInv_vec
	    );
#endif

#endif
	
	if( ddtype_name == Tcalc_name ){ 
	  // double - double 
	  //  std::cout<<" double-double BlockJacobi "<<std::endl;

	  auto *L = Lmatrix_mat.data(); 
	  auto *U = Umatrix_mat.data(); 
	  auto *z = x_.data();
	  auto *r = b_.data();
	  auto *y = work.data();
	  auto *scalingInv_vec = scalingInv_vec_.data();

	  double *L_hi = (double *) Lmatrix_mat.data_hi(); 
	  double *L_lo = (double *) Lmatrix_mat.data_lo(); 
	  double *U_hi = (double *) Umatrix_mat.data_hi(); 
	  double *U_lo = (double *) Umatrix_mat.data_lo(); 
	  double *z_hi = (double *) x_.data_hi();
	  double *z_lo = (double *) x_.data_lo();
	  double *r_hi = (double *) b_.data_hi();
	  double *r_lo = (double *) b_.data_lo();
	  double *y_hi = (double *) work.data_hi();
	  double *y_lo = (double *) work.data_lo();

	  if( ddtype_name ==  Tprecon_mat_name){ 
	    if( ddtype_name == Tvec_name){ 
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : dd
		// Tvec : dd
		// Tprecon_vec_name : dd
		ddblockjacobi(
			      q_,
			      N,
			      r_hi,r_lo,
			      y_hi,y_lo,
			      z_hi,z_lo,
			      L_hi,L_lo,
			      Lmatrix_offset,nndL,
			      U_hi,U_lo,
			      Umatrix_offset,nndU,	      
			      scalingInv_vec);

	      }else{
		// Tprecon_mat : dd
		// Tvec : dd
		// Tprecon_vec_name : d
 		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y,
		    z_hi,z_lo,
		    L_hi,L_lo,
		    Lmatrix_offset,nndL,
		    U_hi,U_lo,
		    Umatrix_offset,nndU,
			      scalingInv_vec);
	      }
	    }else{
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : dd
		// Tvec : d
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y_hi,y_lo,
		    z,
		    L_hi,L_lo,
		    Lmatrix_offset,nndL,
		    U_hi,U_lo,
		    Umatrix_offset,nndU,
		    scalingInv_vec);

	      }else{
		// Tprecon_mat : dd
		// Tvec : d
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y,
		    z,
		    L_hi,L_lo,
		    Lmatrix_offset,nndL,
		    U_hi,U_lo,
		    Umatrix_offset,nndU,
		    scalingInv_vec);

	      }
	    }
	  }else{
	    if( ddtype_name == Tvec_name){ 
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : d
		// Tvec : dd
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y_hi,y_lo,
		    z_hi,z_lo,
		    L,
		    Lmatrix_offset,nndL,
		    U,
		    Umatrix_offset,nndU,
		    scalingInv_vec);

	      }else{
		// Tprecon_mat : d
		// Tvec : dd
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r_hi,r_lo,
		    y,
		    z_hi,z_lo,
		    L,
		    Lmatrix_offset,nndL,
		    U,
		    Umatrix_offset,nndU,
		    scalingInv_vec);

	      }
	    }else{
	      if( ddtype_name ==  Tprecon_vec_name){ 
		// Tprecon_mat : d
		// Tvec : d
		// Tprecon_vec_name : dd
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y_hi,y_lo,
		    z,
		    L,
		    Lmatrix_offset,nndL,
		    U,
		    Umatrix_offset,nndU,
		    scalingInv_vec);
	      }else{
		//		std::cout<<" double all "<<std::endl;
		// Tprecon_mat : d
		// Tvec : d
		// Tprecon_vec_name : d
		ddblockjacobi<
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type
		  >(			      
		    q_,
		    N,
		    r,
		    y,
		    z,
		    L,
		    Lmatrix_offset,nndL,
		    U,
		    Umatrix_offset,nndU,
		    scalingInv_vec);
		return ;
	      }
	    }

	  }
	}else{

	  auto *L = Lmatrix_mat.data(); 
	  auto *U = Umatrix_mat.data(); 
	  auto *z = x_.data();
	  auto *r = b_.data();
	  auto *y = work.data();
	  auto *scalingInv_vec = scalingInv_vec_.data();

	  blockjacobi<
	    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tvec) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_vec) >::type,
	    typename std::result_of<parcel::parcel_type_to_baseType(Tcalc) >::type
	    >(
	      q_,
	      N,
	      r,
	      y,
	      z,
	      L,
	      Lmatrix_offset,
	      nndL,
	      U,
	      Umatrix_offset,
	      nndU,
	      scalingInv_vec
	      );
	}

     return ;

   }

    void solver(
		vector<Tvec> &b_,
		vector<Tvec> &x_
		){

      matrix<Tprecon_mat> &Lmatrix_matrix = blockjacobi_buffer_.Lmatrix();
      matrix<Tprecon_mat> &Umatrix_matrix = blockjacobi_buffer_.Umatrix();

      size_t matrix_type = Lmatrix_matrix.matrix_type();      
      if(matrix_type == CRS_MATRIX){
	parcel::crsMat<Tprecon_mat> &Lmatrix_crsMat = *Lmatrix_matrix.crsMat_data();
	parcel::crsMat<Tprecon_mat> &Umatrix_crsMat = *Umatrix_matrix.crsMat_data();
	solver_main(
		    b_,x_,
		    Lmatrix_crsMat,
		    Umatrix_crsMat);
      } // end CRS_MATRIX
      if(matrix_type == DIA_MATRIX){
	parcel::diaMat<Tprecon_mat> &Lmatrix_diaMat = *Lmatrix_matrix.diaMat_data();
	parcel::diaMat<Tprecon_mat> &Umatrix_diaMat = *Umatrix_matrix.diaMat_data();
	solver_main(
		    b_,x_,
		    Lmatrix_diaMat,
		    Umatrix_diaMat);
      } // end DIA_MATRIX

    }

    void init(
	      crsMat<Tmat> &crsMat,
	      PARCEL_INT BLOCK_JACOBI_NUMBLOCK,
	      bool SCALING_MATRIX,
	      bool SCALING_VECTOR
	      ){
      BLOCK_JACOBI_NUMBLOCK_ = BLOCK_JACOBI_NUMBLOCK;
      set_SCALING_MATRIX(SCALING_MATRIX);
      set_SCALING_VECTOR(SCALING_VECTOR);

      resize_blocksize(crsMat,BLOCK_JACOBI_NUMBLOCK);
      get_blockjacobi_nnzLnnzU(crsMat);
      resize_buffer(crsMat);
      get_blockjacobi_LUmatrix(crsMat);
    }

    void init(
	      diaMat<Tmat> &diaMat,
	      PARCEL_INT BLOCK_JACOBI_NUMBLOCK,
	      bool SCALING_MATRIX,
	      bool SCALING_VECTOR
	      ){
      BLOCK_JACOBI_NUMBLOCK_ = BLOCK_JACOBI_NUMBLOCK;
      set_SCALING_MATRIX(SCALING_MATRIX);
      set_SCALING_VECTOR(SCALING_VECTOR);

      resize_blocksize(diaMat,BLOCK_JACOBI_NUMBLOCK);
      get_blockjacobi_nndLnndU(diaMat);
      get_blockjacobi_LUmatrix(diaMat);
    }

  };

}

#endif
