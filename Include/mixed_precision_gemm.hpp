#ifndef MIXED_PRECISON_GEMM_HPP_
#define MIXED_PRECISON_GEMM_HPP_



template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemm(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  PARCEL_INT ldA,
	  TdataMatB *B, // k行n列
	  PARCEL_INT ldB,
	  TdataMatC *C, // m行n列
	  PARCEL_INT ldC){      

  // 並列軸は出力先のC行列要素数
  size_t MAX   = m*n;
#if _BLOCKING_LOOP_

  // #if _A64FX_HIPSYCL_
#if _A64FX_HIPSYCL_NO  
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = m*n;
  const size_t nlocalblock = 1;
#endif
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){

#endif
      // C[i_c,j_c]
      const PARCEL_INT i_c = i%m;
      const PARCEL_INT j_c = i/m;

      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_ = 0.0;

      for(PARCEL_INT k_=0;k_<k;k_++){
	const PARCEL_INT i_a = i_c;
	const PARCEL_INT j_a = k_;
	const PARCEL_INT i_b = k_;
	const PARCEL_INT j_b = j_c;
	const Tcalc A_ = A[i_a + j_a*ldA];
	const Tcalc B_ = B[i_b + j_b*ldB];
	AB_ = AB_ + A_ * B_;
      }
      Tcalc C_ = C[i_c + j_c*ldC];
      C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

// C[m,n] = alpha * (A[m,k])^T * B[k,n] + beta * C[m,n]
// B基準ループ : AB[m,n] = A[k,m] * B[k,n]
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTN(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  PARCEL_INT ldA,
	  TdataMatB *B, // k行n列
	  PARCEL_INT ldB,
	  TdataMatC *C, // m行n列
	  PARCEL_INT ldC){      

      if( (m == 4) && (n%2==1) ){
	// 並列軸は出力先のC行列要素数
	size_t MAX   = m*n;
#if 1
	//	const size_t kloopblock = k; // 配列サイズnのブロックサイズ
	const size_t kloopblock = 480; // 配列サイズnのブロックサイズ
	const size_t work_group_size = 1;
	const size_t num_work_groups = 24;
	const size_t num_work_items = num_work_groups*work_group_size;
	Tcalc outbuf_[ m*n * num_work_items ];
	sycl::buffer<Tcalc> outbuf(outbuf_, m*n * num_work_items );

	q_[0].submit([&] (sycl::handler &h){
		       auto scratch = outbuf.get_access(h);
		       h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
				      [=](sycl::nd_item<1> item) {
					const size_t global_id = item.get_global_id(0);
					const size_t group_id  = item.get_group(0);	     
					const size_t local_id  = item.get_local_id(0);
					
					for(PARCEL_INT i=0;i<n;i++){
					  const Tcalc beta_ = beta[0];
					  Tcalc C1_ = C[0 + i*ldC];
					  Tcalc C2_ = C[1 + i*ldC];
					  Tcalc C3_ = C[2 + i*ldC];
					  Tcalc C4_ = C[3 + i*ldC];
					  scratch[0+i*ldC +global_id*m*n] = global_id==0 ? beta_*C1_ : (Tcalc) 0.0;
					  scratch[1+i*ldC +global_id*m*n] = global_id==0 ? beta_*C2_ : (Tcalc) 0.0;
					  scratch[2+i*ldC +global_id*m*n] = global_id==0 ? beta_*C3_ : (Tcalc) 0.0;
					  scratch[3+i*ldC +global_id*m*n] = global_id==0 ? beta_*C4_ : (Tcalc) 0.0;
					}

					for(PARCEL_INT kk_=group_id*kloopblock;kk_<k;kk_=kk_+kloopblock*num_work_groups){
					  PARCEL_INT kmax_ = (kk_ + kloopblock) > k ? k : (kk_ + kloopblock);
					  const Tcalc alpha_ = alpha[0];
#if 1
					  {
					    PARCEL_INT i=0;
					    Tcalc AB11_ = 0.0;
					    Tcalc AB21_ = 0.0;
					    Tcalc AB31_ = 0.0;
					    Tcalc AB41_ = 0.0;
#pragma omp simd
					    for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      const Tcalc A1_ = A[k_ + 0*ldA];
					      const Tcalc A2_ = A[k_ + 1*ldA];
					      const Tcalc A3_ = A[k_ + 2*ldA];
					      const Tcalc A4_ = A[k_ + 3*ldA];
					      const Tcalc B1_ = B[k_ + i*ldB];					      
					      AB11_ = AB11_ + A1_ * B1_;
					      AB21_ = AB21_ + A2_ * B1_;
					      AB31_ = AB31_ + A3_ * B1_;
					      AB41_ = AB41_ + A4_ * B1_;
					    }
					    // Tcalc C_ = C[i_c + j_c*ldC];
					    // C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;
					    scratch[0+i*ldC +global_id*m*n] = scratch[0+i*ldC +global_id*m*n] + alpha_*AB11_;
					    scratch[1+i*ldC +global_id*m*n] = scratch[1+i*ldC +global_id*m*n] + alpha_*AB21_;
					    scratch[2+i*ldC +global_id*m*n] = scratch[2+i*ldC +global_id*m*n] + alpha_*AB31_;
					    scratch[3+i*ldC +global_id*m*n] = scratch[3+i*ldC +global_id*m*n] + alpha_*AB41_;

					  }
					  for(PARCEL_INT i=1;i<n;i=i+2){
					    // C[i_c,j_c]
					    Tcalc AB11_ = 0.0;
					    Tcalc AB21_ = 0.0;
					    Tcalc AB31_ = 0.0;
					    Tcalc AB41_ = 0.0;
					    Tcalc AB12_ = 0.0;
					    Tcalc AB22_ = 0.0;
					    Tcalc AB32_ = 0.0;
					    Tcalc AB42_ = 0.0;
#pragma omp simd
					    for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      const Tcalc A1_ = A[k_ + 0*ldA];
					      const Tcalc A2_ = A[k_ + 1*ldA];
					      const Tcalc A3_ = A[k_ + 2*ldA];
					      const Tcalc A4_ = A[k_ + 3*ldA];
					      const Tcalc B1_ = B[k_ + i*ldB];					      
					      const Tcalc B2_ = B[k_ + (i+1)*ldB];					      
					      AB11_ = AB11_ + A1_ * B1_;
					      AB21_ = AB21_ + A2_ * B1_;
					      AB31_ = AB31_ + A3_ * B1_;
					      AB41_ = AB41_ + A4_ * B1_;

					      AB12_ = AB12_ + A1_ * B2_;
					      AB22_ = AB22_ + A2_ * B2_;
					      AB32_ = AB32_ + A3_ * B2_;
					      AB42_ = AB42_ + A4_ * B2_;
					    }
					    // Tcalc C_ = C[i_c + j_c*ldC];
					    // C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;
					    scratch[0+i*ldC +global_id*m*n] = scratch[0+i*ldC +global_id*m*n] + alpha_*AB11_;
					    scratch[1+i*ldC +global_id*m*n] = scratch[1+i*ldC +global_id*m*n] + alpha_*AB21_;
					    scratch[2+i*ldC +global_id*m*n] = scratch[2+i*ldC +global_id*m*n] + alpha_*AB31_;
					    scratch[3+i*ldC +global_id*m*n] = scratch[3+i*ldC +global_id*m*n] + alpha_*AB41_;

					    scratch[0+(i+1)*ldC +global_id*m*n] = scratch[0+(i+1)*ldC +global_id*m*n] + alpha_*AB12_;
					    scratch[1+(i+1)*ldC +global_id*m*n] = scratch[1+(i+1)*ldC +global_id*m*n] + alpha_*AB22_;
					    scratch[2+(i+1)*ldC +global_id*m*n] = scratch[2+(i+1)*ldC +global_id*m*n] + alpha_*AB32_;
					    scratch[3+(i+1)*ldC +global_id*m*n] = scratch[3+(i+1)*ldC +global_id*m*n] + alpha_*AB42_;
					    
					  }
#else					  
					  for(PARCEL_INT i=0;i<n;i++){
					    // C[i_c,j_c]
					    Tcalc AB11_ = 0.0;
					    Tcalc AB21_ = 0.0;
					    Tcalc AB31_ = 0.0;
					    Tcalc AB41_ = 0.0;
#pragma omp simd
					    for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      const Tcalc A1_ = A[k_ + 0*ldA];
					      const Tcalc A2_ = A[k_ + 1*ldA];
					      const Tcalc A3_ = A[k_ + 2*ldA];
					      const Tcalc A4_ = A[k_ + 3*ldA];
					      const Tcalc B1_ = B[k_ + i*ldB];					      
					      AB11_ = AB11_ + A1_ * B1_;
					      AB21_ = AB21_ + A2_ * B1_;
					      AB31_ = AB31_ + A3_ * B1_;
					      AB41_ = AB41_ + A4_ * B1_;
					    }
					    // Tcalc C_ = C[i_c + j_c*ldC];
					    // C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;
					    scratch[0+i*ldC +global_id*m*n] = scratch[0+i*ldC +global_id*m*n] + alpha_*AB11_;
					    scratch[1+i*ldC +global_id*m*n] = scratch[1+i*ldC +global_id*m*n] + alpha_*AB21_;
					    scratch[2+i*ldC +global_id*m*n] = scratch[2+i*ldC +global_id*m*n] + alpha_*AB31_;
					    scratch[3+i*ldC +global_id*m*n] = scratch[3+i*ldC +global_id*m*n] + alpha_*AB41_;
					  }
#endif					  
					}
					
				      });
		     });
	q_[0].wait();
	for(int j=1;j<num_work_items;j++){	     
	  for(int i=0;i<m*n;i++){	     
	    outbuf_[i+0*m*n] =outbuf_[i+0*m*n] + outbuf_[i+j*m*n] ;
	  }
	}
	q_[0].wait();
	q_[0].submit([&] (sycl::handler &h){
		       auto scratch = outbuf.get_access(h);
		       h.parallel_for(sycl::nd_range<1>(1, 1),
				      [=](sycl::nd_item<1> item) {
					for(PARCEL_INT i=0;i<m*n;i++){
					  const PARCEL_INT i_c = i%m;
					  const PARCEL_INT j_c = i/m;
					  C[i_c + j_c*ldC] = scratch[i+0*m*n];
					}
				      });	  
		     });	  
	q_[0].wait();
#else
	q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
						 // C[i_c,j_c]
						 const PARCEL_INT i_c = i%m;
						 const PARCEL_INT j_c = i/m;

						 const Tcalc alpha_ = alpha[0];
						 const Tcalc beta_  = beta[0];
						 Tcalc AB_ = 0.0;

						 for(PARCEL_INT k_=0;k_<k;k_++){
						   const PARCEL_INT i_a = k_;
						   const PARCEL_INT j_a = i_c;
						   const PARCEL_INT i_b = k_;
						   const PARCEL_INT j_b = j_c;
						   const Tcalc A_ = A[i_a + j_a*ldA];
						   const Tcalc B_ = B[i_b + j_b*ldB];
						   AB_ = AB_ + A_ * B_;
						 }
						 Tcalc C_ = C[i_c + j_c*ldC];
						 C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;
					       });
	
#endif	
	return ;
      }
   // 並列軸は出力先のC行列要素数
  size_t MAX   = m*n;
#if _BLOCKING_LOOP_

  // #if _A64FX_HIPSYCL_
#if _A64FX_HIPSYCL_NO_  
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = m*n;
  const size_t nlocalblock = 1;
#endif
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){

#endif
      // C[i_c,j_c]
      const PARCEL_INT i_c = i%m;
      const PARCEL_INT j_c = i/m;

      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_ = 0.0;

      for(PARCEL_INT k_=0;k_<k;k_++){
	const PARCEL_INT i_a = k_;
	const PARCEL_INT j_a = i_c;
	const PARCEL_INT i_b = k_;
	const PARCEL_INT j_b = j_c;
	const Tcalc A_ = A[i_a + j_a*ldA];
	const Tcalc B_ = B[i_b + j_b*ldB];
	AB_ = AB_ + A_ * B_;
      }
      Tcalc C_ = C[i_c + j_c*ldC];
      C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

// Cは上三角部分のみ計算して下三角部分を複製 (m==n)
// C[m,n] = alpha * (A[m,k])^T * B[k,n] + beta * C[m,n]
// B基準ループ : AB[m,n] = A[k,m] * B[k,n]
// 

#if 0
// #ifdef __CUDACC__
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTNsym(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  PARCEL_INT ldA,
	  TdataMatB *B, // k行n列
	  PARCEL_INT ldB,
	  TdataMatC *C, // m行n列
	  PARCEL_INT ldC){      

  const size_t mm_num = ((m+1)*m)/2; // Cは正方行列(m==n)
  const size_t nblock = 16384 > k ? k : 16384; // 配列サイズnのブロックサイズ
  //  const size_t mm_group_size = (1 + k / nblock) > 512? 512 : (1 + n / nblock) ; // 下三角領域を計算するグループ多重度
  //  const size_t mm_group_size = (1 + k / nblock) > 128? 128 : (1 + n / nblock) ; // 下三角領域を計算するグループ多重度
  const size_t mm_group_size = 1; // 下三角領域を計算するグループ多重度
    
  const size_t ngroup_block = nblock * mm_group_size;
  //  const size_t work_group_size = 1024; // simd lengh
  const size_t work_group_size = 1024; // simd lengh
  //    const size_t work_group_size = 512; // simd lengh
  //  const size_t work_group_size = 256; // simd lengh
  //  const size_t work_group_size = 32; // simd lengh
  const size_t num_work_groups = mm_num * mm_group_size; 
  const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
  
  //  Tcalc outbuf_[num_work_groups];
  //  sycl::buffer<Tcalc> outbuf(outbuf_,num_work_groups);  
  Tcalc outbuf_[m*m*mm_group_size];
  
  //  sycl::buffer<Tcalc> outbuf(outbuf_,m*m*mm_group_size);
  Tcalc * out = sycl::malloc_device<Tcalc>(m*m*mm_group_size,q_[0]);

  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	Tcalc,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch(work_group_size, h); 
      //      auto out = outbuf.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
             const size_t global_id = item.get_global_id(0);
	     const size_t batch_id  = global_id/work_group_size/mm_num;
	     const size_t local_id  = item.get_local_id(0);
	     const size_t group_id  = item.get_group(0);	     

	     // 列番号
	     int icol = 0;
	     {
	       int mm_ = group_id;
	       for(int i=0;i<mm_num;i++){
		 mm_ = mm_ - i;
		 if(mm_<0){
		   icol = i-1;	
		   break ;
		 }
	       }
	     }
    
	     // 行番号
	     int irow = 0;
	     {
	       int mm_ = group_id;
	       for(int i=0;i<mm_num;i++){
		 mm_ = mm_ - 1;
		 if(mm_<=0){
		   irow = 0;	
		   for(int j=0;j<=icol;j++){
		     irow = irow + j;
		   } 
		   irow = (group_id - irow)%m;
		   break ;
		 }      
	       }
	     }
	     
      // C[i_c,j_c]
      const PARCEL_INT i_c = irow;
      const PARCEL_INT j_c = icol;
      scratch[local_id] = batch_id == 0 ? C[i_c + j_c*ldC] : (TdataMatC)0.0;

      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_ = 0.0;

      const int kk0 = ngroup_block*batch_id;
      for(int kk = kk0; kk<k; kk = kk + ngroup_block*mm_group_size){      
	const int ks = kk + local_id;
	int ke = (kk + ngroup_block) > k ? k : (kk + ngroup_block);
	for(int k_ = ks; k_<ke; k_ = k_ + work_group_size){
	  //	for(PARCEL_INT k_=0;k_<k;k_++){
	  const PARCEL_INT i_a = k_;
	  const PARCEL_INT j_a = i_c;
	  const PARCEL_INT i_b = k_;
	  const PARCEL_INT j_b = j_c;
	  const Tcalc A_ = A[i_a + j_a*ldA];
	  const Tcalc B_ = B[i_b + j_b*ldB];
	  AB_ = AB_ + A_ * B_;
	}
      }
      scratch[local_id] += alpha_*AB_;
      // Do a tree reduction on items in work-group
      for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
	item.barrier(sycl::access::fence_space::local_space);
	if (local_id < i){
	  scratch[local_id] += scratch[local_id + i];
	}
      }
      if (local_id == 0) {
	//	out[group_id] = scratch[0];
	out[ i_c + j_c*m + batch_id*m*m] = scratch[0];
      }
     });
    });

  q_[0].wait();

  const size_t bytes = m*m*mm_group_size * sizeof(Tcalc);
  for(size_t i=0;i<m*m*mm_group_size;i++){ outbuf_[i] = i; }
  
  q_[0].memcpy(outbuf_,out,bytes);
  q_[0].wait();
  for(size_t j=1;j<mm_group_size;j++){
    for(size_t i=0;i<m*m;i++){
      outbuf_[i] = outbuf_[i] + outbuf_[i+j*m*m];
    }
  }

  q_[0].memcpy(out,outbuf_,bytes);
  q_[0].wait();
  
  q_[0].submit([&] (sycl::handler &h){
      h.parallel_for(sycl::nd_range<1>(m*m,1),
		     [=](sycl::nd_item<1> item) {
             const size_t global_id = item.get_global_id(0);
	     const PARCEL_INT i_c = global_id%m;
	     const PARCEL_INT j_c = global_id/m;
	     if( i_c <= j_c ){
	       C[i_c + j_c*ldC] = out[ i_c + j_c*m ] ;
	       C[j_c + i_c*ldC] = out[ i_c + j_c*m ] ;
	     }
		     });
	       });
  q_[0].wait();
  
  return ;
  
}
    
#else

template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTNsym(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  PARCEL_INT ldA,
	  TdataMatB *B, // k行n列
	  PARCEL_INT ldB,
	  TdataMatC *C, // m行n列
	  PARCEL_INT ldC){      

  //size_t MAX   = m*n;
  //  const size_t kloopblock = 16384; // 配列サイズnのブロックサイズ
  const size_t kloopblock = 480; // 配列サイズnのブロックサイズ
  // const size_t kloopblock = 1; // 配列サイズnのブロックサイズ
  //  const size_t num_work_items = 24;
  const size_t work_group_size = 1;
  //  const size_t num_work_groups = 24;
  const size_t num_work_groups = 32;
  const size_t num_work_items = num_work_groups*work_group_size;
  const size_t mm = m*m; // Cは正方行列(m==n)

  // 正方行列サイズ * 並列数
  Tcalc outbuf_[ mm * num_work_items ];
  sycl::buffer<Tcalc> outbuf(outbuf_, mm * num_work_items );
  q_[0].submit([&] (sycl::handler &h){
		 auto scratch = outbuf.get_access(h);
		 h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
				[=](sycl::nd_item<1> item) {
				  const size_t global_id = item.get_global_id(0);
				  const size_t group_id  = item.get_group(0);	     
				  const size_t local_id  = item.get_local_id(0);

				  for(int irow=0;irow<m;irow++){	     
				    for(int icol=irow;icol<m;icol++){
				      const Tcalc beta_  = beta[0];
				      const PARCEL_INT i_c = irow;
				      const PARCEL_INT j_c = icol;
				      Tcalc C_ = C[i_c + j_c*ldC];					
				      scratch[irow+icol*m+global_id*mm] = global_id==0 ? beta_*C_ : (Tcalc)0.0;
				    }
				  }
				  
				  // for(PARCEL_INT kk_=0;kk_<k;kk_=kk_+kloopblock){
#if 1
				  for(PARCEL_INT kk_=group_id*kloopblock;kk_<k;kk_=kk_+kloopblock*num_work_groups){
#else				  
				  for(PARCEL_INT kk_=global_id*kloopblock;kk_<k;kk_=kk_+kloopblock*num_work_items){
#endif				    
				    PARCEL_INT kmax_ = (kk_ + kloopblock) > k ? k : (kk_ + kloopblock);

#if 1
				    if( m%2 == 0 ){
#if 1
				      // block loop				      
				      const PARCEL_INT rowblock = 2;
				      const PARCEL_INT colblock = 2;
				      for(PARCEL_INT irow_=0;irow_<m;irow_=irow_+rowblock){
					PARCEL_INT irow_max = (irow_+rowblock) > m ? m : (irow_+rowblock);
					for(PARCEL_INT icol_=irow_;icol_<m;icol_=icol_+colblock){
					  PARCEL_INT icol_max = (icol_+colblock) > m ? m : (icol_+colblock);
				      
					  // for(PARCEL_INT irow=0;irow<m;irow=irow+2){
					  for(PARCEL_INT irow=irow_;irow<irow_max;irow=irow+2){
					    const PARCEL_INT i1 = irow;
					    const PARCEL_INT i2 = irow+1;				      
					    //					for(int icol=irow;icol<m;icol=icol+2){	     
					    for(PARCEL_INT icol=icol_;icol<icol_max;icol=icol+2){
					      const PARCEL_INT j1 = icol;
					      const PARCEL_INT j2 = icol+1;					
					      Tcalc AB11_ = 0.0;	
					      Tcalc AB12_ = 0.0;	
					      Tcalc AB21_ = 0.0;	
					      Tcalc AB22_ = 0.0;
					      // ここのループを改良したい
					      // for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					      // for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      // for(PARCEL_INT k_=kk_+global_id;k_<kmax_;k_=k_+num_work_items){
#if 1
#pragma omp simd
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
#else
#pragma omp simd
						for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
						//						for(PARCEL_INT k_=kk_;k_<kmax_;k_=k_+2){
#endif						
						const Tcalc A1_ = A[k_ + i1*ldA];
						const Tcalc A2_ = A[k_ + i2*ldA];
						const Tcalc B1_ = B[k_ + j1*ldB];
						const Tcalc B2_ = B[k_ + j2*ldB];

						// const Tcalc A1_2_ = A[k_+1 + i1*ldA];
						// const Tcalc A2_2_ = A[k_+1 + i2*ldA];
						// const Tcalc B1_2_ = B[k_+1 + j1*ldB];
						// const Tcalc B2_2_ = B[k_+1 + j2*ldB];
						
						AB11_ = AB11_ + A1_ * B1_;// + A1_2_ * B1_2_;
						AB12_ = AB12_ + A1_ * B2_;// + A1_2_ * B2_2_;
						AB21_ = AB21_ + A2_ * B1_;// + A2_2_ * B1_2_;
						AB22_ = AB22_ + A2_ * B2_;// + A2_2_ * B2_2_;
					      }
					      const Tcalc alpha_ = alpha[0];					
					      scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					      scratch[i1+j2*m+global_id*mm] += alpha_*AB12_;
					      scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					      scratch[i2+j2*m+global_id*mm] += alpha_*AB22_;
					    }
					  }
					}
				      }
#else
				      // block loop				      
				      const PARCEL_INT rowblock = 2;
				      const PARCEL_INT colblock = 2;
				      for(PARCEL_INT irow_=0;irow_<m;irow_=irow_+rowblock){
					PARCEL_INT irow_max = (irow_+rowblock) > m ? m : (irow_+rowblock);
					for(PARCEL_INT icol_=irow_;icol_<m;icol_=icol_+colblock){
					  PARCEL_INT icol_max = (icol_+colblock) > m ? m : (icol_+colblock);
				      
					  // for(PARCEL_INT irow=0;irow<m;irow=irow+2){
					  for(PARCEL_INT irow=irow_;irow<irow_max;irow=irow+2){
					    const PARCEL_INT i1 = irow;
					    const PARCEL_INT i2 = irow+1;				      
					    //					for(int icol=irow;icol<m;icol=icol+2){	     
					    for(PARCEL_INT icol=icol_;icol<icol_max;icol=icol+2){
					      const PARCEL_INT j1 = icol;
					      const PARCEL_INT j2 = icol+1;					
					      Tcalc AB11_ = 0.0;	
					      Tcalc AB12_ = 0.0;	
					      Tcalc AB21_ = 0.0;	
					      Tcalc AB22_ = 0.0;	
#pragma omp simd
					      // for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
						const Tcalc A1_ = A[k_ + i1*ldA];
						const Tcalc A2_ = A[k_ + i2*ldA];
						const Tcalc B1_ = B[k_ + j1*ldB];
						const Tcalc B2_ = B[k_ + j2*ldB];
						AB11_ = AB11_ + A1_ * B1_;
						AB12_ = AB12_ + A1_ * B2_;
						AB21_ = AB21_ + A2_ * B1_;
						AB22_ = AB22_ + A2_ * B2_;
					      }
					      const Tcalc alpha_ = alpha[0];					
					      scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					      scratch[i1+j2*m+global_id*mm] += alpha_*AB12_;
					      scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					      scratch[i2+j2*m+global_id*mm] += alpha_*AB22_;
					    }
					  }
					}
				      }
#endif				      
				    }else{
				      for(PARCEL_INT irow=0;irow<m-1;irow=irow+2){					
					const PARCEL_INT i1 = irow;
					const PARCEL_INT i2 = irow+1;				      
					for(int icol=irow;icol<m-1;icol=icol+2){	     
					  const PARCEL_INT j1 = icol;
					  const PARCEL_INT j2 = icol+1;					
					  Tcalc AB11_ = 0.0;	
					  Tcalc AB12_ = 0.0;	
					  Tcalc AB21_ = 0.0;	
					  Tcalc AB22_ = 0.0;	
					   #pragma omp simd
					  //					  for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					    const Tcalc A1_ = A[k_ + i1*ldA];
					    const Tcalc A2_ = A[k_ + i2*ldA];
					    const Tcalc B1_ = B[k_ + j1*ldB];
					    const Tcalc B2_ = B[k_ + j2*ldB];
					    AB11_ = AB11_ + A1_ * B1_;
					    AB12_ = AB12_ + A1_ * B2_;
					    AB21_ = AB21_ + A2_ * B1_;
					    AB22_ = AB22_ + A2_ * B2_;
					  }
					  const Tcalc alpha_ = alpha[0];					
					  scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					  scratch[i1+j2*m+global_id*mm] += alpha_*AB12_;
					  scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					  scratch[i2+j2*m+global_id*mm] += alpha_*AB22_;
					}
					{
					  const PARCEL_INT icol = m-1;
					  const PARCEL_INT j1 = icol;
					  Tcalc AB11_ = 0.0;	
					  Tcalc AB21_ = 0.0;	
					   #pragma omp simd
					  //					  for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					    const Tcalc A1_ = A[k_ + i1*ldA];
					    const Tcalc A2_ = A[k_ + i2*ldA];
					    const Tcalc B1_ = B[k_ + j1*ldB];
					    AB11_ = AB11_ + A1_ * B1_;
					    AB21_ = AB21_ + A2_ * B1_;
					  }
					  const Tcalc alpha_ = alpha[0];					
					  scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					  scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					}
				      }
				      {
					const PARCEL_INT irow = m-1;
					const PARCEL_INT icol = m-1;
					const PARCEL_INT i1 = irow;
					const PARCEL_INT j1 = icol;
					Tcalc AB11_ = 0.0;	
					 #pragma omp simd
					//					for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					  const Tcalc A1_ = A[k_ + i1*ldA];
					  const Tcalc B1_ = B[k_ + j1*ldB];
					  AB11_ = AB11_ + A1_ * B1_;
					}
					const Tcalc alpha_ = alpha[0];					
					scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
				      }
				    }
				    
#else				    
				    for(int irow=0;irow<m;irow++){	     
				      for(int icol=irow;icol<m;icol++){	     
					// C[i_c,j_c]
					const PARCEL_INT i_c = irow;
					const PARCEL_INT j_c = icol;
					
					Tcalc AB_ = 0.0;	
#pragma omp simd
					//					for(PARCEL_INT k_=kk_;k_<kmax_;k_++){
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					  const PARCEL_INT i_a = k_;
					  const PARCEL_INT j_a = i_c;
					  const PARCEL_INT i_b = k_;
					  const PARCEL_INT j_b = j_c;
					  const Tcalc A_ = A[i_a + j_a*ldA];
					  const Tcalc B_ = B[i_b + j_b*ldB];
					  AB_ = AB_ + A_ * B_;
					}
					const Tcalc alpha_ = alpha[0];					
					scratch[irow+icol*m+global_id*mm] = scratch[irow+icol*m+global_id*mm]+alpha_*AB_;
				      }
				    }
#endif				    
				  }
				
				});
	       });
  q_[0].wait();
  for(int j=1;j<num_work_items;j++){	     
    for(int i=0;i<mm;i++){	     
      outbuf_[i+0*mm] =outbuf_[i+0*mm] + outbuf_[i+j*mm] ;
    }
  }
  q_[0].wait();
  q_[0].submit([&] (sycl::handler &h){
		 auto scratch = outbuf.get_access(h);
		 h.parallel_for(sycl::nd_range<1>(1, 1),
				[=](sycl::nd_item<1> item) {
				  for(int irow=0;irow<m;irow++){	     
				    for(int icol=irow;icol<m;icol++){
					const PARCEL_INT i_c = irow;
					const PARCEL_INT j_c = icol;
					C[i_c + j_c*ldC] = scratch[irow+icol*m+0*mm] ;					
					C[j_c + i_c*ldC] = scratch[irow+icol*m+0*mm] ;					
				      }
				    }
				});
	       });
  q_[0].wait();

}

#endif
    
#endif
