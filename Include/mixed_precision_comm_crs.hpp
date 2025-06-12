#ifndef MIXED_PRECISON_COMM_HPP_
#define MIXED_PRECISON_COMM_HPP_

template<class Tdata>
void SpMV_comm_pack(
		    sycl::queue *q_,
		    PARCEL_INT nstart,
		    PARCEL_INT nsend,
		    PARCEL_INT *nsend_gcol_list,
		    Tdata *x,
		    Tdata *send_buff
		    ){
  size_t MAX = nsend;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
					   PARCEL_INT icol = nsend_gcol_list[i] - nstart;
					   send_buff[i] = x[icol];
    });
  return ;
}

template<class Tdata>
void SpMV_comm_unpack(
		    sycl::queue *q_,
		    PARCEL_INT nblock,
		    PARCEL_INT nrecv,
		    PARCEL_INT recv_ptr,
		    Tdata *x,
		    Tdata *recv_buff
		    ){
  size_t MAX = nrecv;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
					   x[nblock+i+recv_ptr] = recv_buff[i];
    });
  return ;
}


#endif
