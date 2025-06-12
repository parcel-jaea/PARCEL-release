#ifndef PARCEL_TYPE_HPP_
#define PARCEL_TYPE_HPP_

// using half_float::half;
// using parcel_half = parcel::floatingtype<half>;

using parcel_double       = parcel::floatingtype<double>;
using parcel_float        = parcel::floatingtype<float>;
using parcel_longdouble   = parcel::floatingtype<long double>;
// using parcel_Float128   = parcel::floatingtype<_Float128>;
//@  using parcel_float128   = parcel::floatingtype<__float128>;
// using parcel_fp16         = parcel::floatingtype<sycl::float16>;
// using parcel_bf16         = parcel::floatingtype<__nv_bfloat16>;
// using parcel_fp16         = parcel::floatingtype<sycl::half>;

using parcel_fp16 = parcel::floatingtype<sycl::half>; // dpc++利用可能

// V100で利用不可. : Bfloat16
// using parcel_bf16 = parcel::floatingtype<sycl::ext::oneapi::experimental::bfloat16>;

// hipSYCLならgccのバージョンにより利用可能
// using parcel_fp16 = parcel::floatingtype<_Float16>;


using parcel_doubledouble = parcel::doubledouble;

namespace parcel{

  const auto ddtype_name = typeid(parcel_doubledouble).name();

  struct parcel_type_to_baseType{
    // template<typename T>T ret_T(T a) { 
    //   return a;
    // }
    template<class T> auto operator ()(T a) { 
      auto a_ = a.data();
      return a_[0];
    }

#if 0
    template<class T> auto operator ()(T a) { 
      auto a_ = a.data();
      return a_[0];
    }
#endif

  };

}

// using _longdouble = std::invoke_result<parcel::parcel_type_to_baseType,parcel_longdouble>::type;

// #define parcel_datatype( _parcel_type_ ) std::invoke_result<parcel::parcel_type_to_baseType, _parcel_type_ >::type

#if 0
namespace parcel{
  class parcel_type{
    // auto parcel_float_name        = typeid(parcel_float).name();
    // auto parcel_longdouble_name   = typeid(parcel_longdouble).name();
    // auto parcel_fp16_name         = typeid(parcel_fp16).name();
    // auto parcel_doubledouble_name = typeid(parcel_doubledouble).name();
  public:
    parcel_type(){};
    ~parcel_type(){};
    //    auto retType(Ttype d){
    //      auto dname = typeid(d).name();
    //}
  };

}
#endif

#endif

