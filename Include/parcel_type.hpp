// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_TYPE_HPP_
#define PARCEL_TYPE_HPP_

using parcel_double       = parcel::floatingtype<double>;
using parcel_float        = parcel::floatingtype<float>;
using parcel_longdouble   = parcel::floatingtype<long double>;

using parcel_fp16 = parcel::floatingtype<sycl::half>; // dpc++利用可能


using parcel_doubledouble = parcel::doubledouble;

namespace parcel{

  const auto ddtype_name = typeid(parcel_doubledouble).name();

  struct parcel_type_to_baseType{
    template<class T> auto operator ()(T a) { 
      auto a_ = a.data();
      return a_[0];
    }


  };

}



#endif

