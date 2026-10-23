// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_COMM_HPP_
#define MIXED_PRECISON_COMM_HPP_

// 袖通信の詰め込み・取り出し。添字はランク内の型 PARCEL_LINT。
//   nsend_lcol_list は送る値の自ランク内の番号 (大域列番号 - nstart。set_comm_list で変換済み)。
template<class Tdata>
void SpMV_comm_pack(
		    sycl::queue *q_,
		    PARCEL_LINT nsend,
		    PARCEL_LINT *nsend_lcol_list,
		    Tdata *x,
		    Tdata *send_buff
		    ){
  size_t MAX = nsend;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
					   const PARCEL_LINT icol = nsend_lcol_list[i];
					   send_buff[i] = x[icol];
    });
  return ;
}

template<class Tdata>
void SpMV_comm_unpack(
		    sycl::queue *q_,
		    PARCEL_LINT nblock,
		    PARCEL_LINT nrecv,
		    PARCEL_LINT recv_ptr,
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
