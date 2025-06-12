#ifndef POISSON_HPP_
#define POISSON_HPP_

// #include<quadmath.h>

#include <cstdint>
#include <cmath>
#include "mat.hpp"

template<class T>
class poisson_equation{

public :

  PARCEL_INT get_nnd(){
    return 7;
  }

  void make_matrix(
		   PARCEL_INT n,
		   dia_mat<T> &diaA){

    PARCEL_INT nnd = get_nnd();
    PARCEL_INT nx  = n;
    PARCEL_INT ny  = n;
    PARCEL_INT nz  = n;
    PARCEL_INT ndim = nx*ny*nz;

    diaA.setValue_offset(-nx*ny,0);
    diaA.setValue_offset(-nx   ,1);
    diaA.setValue_offset(-1    ,2);
    diaA.setValue_offset(0     ,3);
    diaA.setValue_offset(1     ,4);
    diaA.setValue_offset(nx    ,5);
    diaA.setValue_offset(nx*ny ,6);
    for(auto i = 0; i < nnd*ndim; i++){
      diaA.setValue_matA(0.0,i);
    }

    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	for(auto ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;

	  PARCEL_INT icw = ix==0      ? -1 : icc - 1;
	  PARCEL_INT ice = ix==(nx-1) ? -1 : icc + 1;

	  PARCEL_INT ics = iy==0      ? -1 : icc - nx;
	  PARCEL_INT icn = iy==(ny-1) ? -1 : icc + nx;

	  PARCEL_INT icb = iz==0      ? -1 : icc - nx*ny;
	  PARCEL_INT ict = iz==(nz-1) ? -1 : icc + nx*ny;

	  T cb = 1.0;
	  T cs = 1.0;
	  T cw = 1.0;
#if 1
	  T cc = -6.0;
	  T ce = 1.0;
	  T cn = 1.0;
	  T ct = 1.0;
#else
	  T ce = 2.4;
	  T cn = 2.3;
	  T ct = 2.2;
	  T cc = -(cb+cs+cw+ce+cn+ct);
#endif
	  

	  if(0<=icb){diaA.setValue_matA(cb,icc + ndim * 0);}
	  if(0<=ics){diaA.setValue_matA(cs,icc + ndim * 1);}
	  if(0<=icw){diaA.setValue_matA(cw,icc + ndim * 2);}
	  if(0<=icc){
	    diaA.setValue_matA(cc,icc + ndim * 3);
	    diaA.setValue_matDiag(cc,icc);
	  }
	  if(0<=ice){diaA.setValue_matA(ce,icc + ndim * 4);}
	  if(0<=icn){diaA.setValue_matA(cn,icc + ndim * 5);}
	  if(0<=ict){diaA.setValue_matA(ct,icc + ndim * 6);}
	 
	}
      }
    }

  }

  PARCEL_INT get_nnz(PARCEL_INT n){
    PARCEL_INT nnz_ = 0;
    PARCEL_INT nx = n;
    PARCEL_INT ny = n;
    PARCEL_INT nz = n;

    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	for(auto ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;

	  PARCEL_INT icw = ix==0      ? -1 : icc - 1;
	  PARCEL_INT ice = ix==(nx-1) ? -1 : icc + 1;

	  PARCEL_INT ics = iy==0      ? -1 : icc - nx;
	  PARCEL_INT icn = iy==(ny-1) ? -1 : icc + nx;

	  PARCEL_INT icb = iz==0      ? -1 : icc - nx*ny;
	  PARCEL_INT ict = iz==(nz-1) ? -1 : icc + nx*ny;

	  if(0<=icb)nnz_ = nnz_ + 1;
	  if(0<=ics)nnz_ = nnz_ + 1;
	  if(0<=icw)nnz_ = nnz_ + 1;
	  if(0<=icc)nnz_ = nnz_ + 1;
	  if(0<=ice)nnz_ = nnz_ + 1;
	  if(0<=icn)nnz_ = nnz_ + 1;
	  if(0<=ict)nnz_ = nnz_ + 1;

	}
      }
    }
    return nnz_;

  }

  void make_matrix(
		   PARCEL_INT n,
		   crs_mat<T> &crsA){

    PARCEL_INT nnz = 0;
    PARCEL_INT nx  = n;
    PARCEL_INT ny  = n;
    PARCEL_INT nz  = n;
    crsA.setValue_crsRow_ptr(0,0);
    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	for(auto ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;

	  PARCEL_INT icw = ix==0      ? -1 : icc - 1;
	  PARCEL_INT ice = ix==(nx-1) ? -1 : icc + 1;

	  PARCEL_INT ics = iy==0      ? -1 : icc - nx;
	  PARCEL_INT icn = iy==(ny-1) ? -1 : icc + nx;

	  PARCEL_INT icb = iz==0      ? -1 : icc - nx*ny;
	  PARCEL_INT ict = iz==(nz-1) ? -1 : icc + nx*ny;

	  T cb = 1.0;
	  T cs = 1.0;
	  T cw = 1.0;
	  T cc = -6.0;
	  T ce = 1.0;
	  T cn = 1.0;
	  T ct = 1.0;

	  crsA.setValue_crsRow_ptr(nnz,icc);
	  if(0<=icb){crsA.setValue_matA(cb,nnz);crsA.setValue_crsCol(icb,nnz);nnz=nnz+1;}
	  if(0<=ics){crsA.setValue_matA(cs,nnz);crsA.setValue_crsCol(ics,nnz);nnz=nnz+1;}
	  if(0<=icw){crsA.setValue_matA(cw,nnz);crsA.setValue_crsCol(icw,nnz);nnz=nnz+1;}
	  if(0<=icc){
	    crsA.setValue_matA(cc,nnz);crsA.setValue_crsCol(icc,nnz);nnz=nnz+1;
	    crsA.setValue_matDiag(cc,icc);
	  }
	  if(0<=ice){crsA.setValue_matA(ce,nnz);crsA.setValue_crsCol(ice,nnz);nnz=nnz+1;}
	  if(0<=icn){crsA.setValue_matA(cn,nnz);crsA.setValue_crsCol(icn,nnz);nnz=nnz+1;}
	  if(0<=ict){crsA.setValue_matA(ct,nnz);crsA.setValue_crsCol(ict,nnz);nnz=nnz+1;}

	  // if(0<=icb){crsA.setValue_matA(cb,icb);}
	  // if(0<=ics){crsA.setValue_matA(cs,ics);}
	  // if(0<=icw){crsA.setValue_matA(cw,icw);}
	  // if(0<=icc){crsA.setValue_matA(cc,icc);}
	  // if(0<=ice){crsA.setValue_matA(ce,ice);}
	  // if(0<=icn){crsA.setValue_matA(cn,icn);}
	  // if(0<=ict){crsA.setValue_matA(ct,ict);}
	 
	}
      }
    }
    crsA.setValue_crsRow_ptr(nnz,nx*ny*nz);
  }

  void make_right_vector(
			 PARCEL_INT n,
			 std::vector<T> &b
			 ){
    PARCEL_INT nx  = n;
    PARCEL_INT ny  = n;
    PARCEL_INT nz  = n;

    T hx = 1.0/(T)(nx+1);
    T hy = 1.0/(T)(ny+1);
    T hz = 1.0/(T)(nz+1);
    T z = 2.0*(hx*hx)*(hy*hy)*(hz*hz);

    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	for(auto ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	  T tmpX = (T)(ix+1) * (T)(ix+1);
	  T tmpY = (T)(iy+1) * (T)(iy+1);
	  T tmpZ = (T)(iz+1) * (T)(iz+1);
	  b[icc] = z *( tmpX*tmpY +tmpY*tmpZ +tmpZ*tmpX );
	}
      }
    }

    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	auto ix = nx - 1;
	PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	b[icc] = b[icc] - (hy*(iy+1))*(hy*(iy+1)) * (hz*(iz+1))*(hz*(iz+1));
      }
    }
    for(auto iz = 0; iz < nz; iz++){
      for(auto ix = 0; ix < nx; ix++){
	auto iy = ny - 1;
	PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	b[icc] = b[icc] - (hx*(ix+1))*(hx*(ix+1)) * (hz*(iz+1))*(hz*(iz+1));
      }
    }
    for(auto iy = 0; iy < ny; iy++){
      for(auto ix = 0; ix < nx; ix++){
	auto iz = nz - 1;
	PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	b[icc] = b[icc] - (hx*(ix+1))*(hx*(ix+1)) * (hy*(iy+1))*(hy*(iy+1));
      }
    }

    for(auto iz = 0; iz < nz; iz++){
      for(auto iy = 0; iy < ny; iy++){
	for(auto ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	  T tmpX = (T)(ix+1) * (T)(ix+1);
	  T tmpY = (T)(iy+1) * (T)(iy+1);
	  T tmpZ = (T)(iz+1) * (T)(iz+1);
	  //	  std::cout<<icc<<" b0 = "<<b[icc]<<std::endl;
	}
      }
    }

  }

  void check_solution(
		      std::vector<T> &x,
		      PARCEL_INT n 
		      ){
    PARCEL_INT nx  = n;
    PARCEL_INT ny  = n;
    PARCEL_INT nz  = n;

    T hx = 1.0/(T)(nx+1);
    T hy = 1.0/(T)(ny+1);
    T hz = 1.0/(T)(nz+1);
    T z  = (hx*hx)*(hy*hy)*(hz*hz);
    double max_error = 0.0;
    double max_exact = 0.0;
    for(PARCEL_INT iz = 0; iz < nz; iz++){
      for(PARCEL_INT iy = 0; iy < ny; iy++){
	for(PARCEL_INT ix = 0; ix < nx; ix++){
	  PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
	  T tmpX = (T)(ix+1) * (T)(ix+1);
	  T tmpY = (T)(iy+1) * (T)(iy+1);
	  T tmpZ = (T)(iz+1) * (T)(iz+1);
	  T x0 = z *( tmpX*tmpY*tmpZ  );
	  T exact = std::abs(x0);
	  T error = std::abs(x[icc] - exact);
	  //	  std::cout<<icc<<" : error = "<<error<<" , "<<x[icc]<<std::endl;
	  //	  std::cout<<ix<<" , "<<iy<<" , "<<iz<<" , "<<" -- "<<exact<<" : "<<x[icc]<<std::endl;
	  if(max_error < error){
	    max_error = error;
	    max_exact = exact;
	  }
	}
      }
    }
    //    std::cout<<nx<<" , "<<ny<<" , "<<nz<<std::endl;

#if 0
      char str_err1[128];
      char str_err2[128];
      quadmath_snprintf(str_err1,128,"%.40Qf",max_error);
      quadmath_snprintf(str_err2,128,"%.40Qf",max_error/max_exact);
      printf(" max(|numerical-exact|)                = %s \n",str_err1);
      printf(" max(|numerical-exact|) / max(|exact|) = %s \n",str_err2);
#else

#if 0
      double error1_ = max_error;
      double error2_ = max_error/max_exact;
      std::cout<<"max(|numerical-exact|)                = "<<error1_<<std::endl;
      std::cout<<"max(|numerical-exact|) / max(|exact|) = "<<error2_<<std::endl;
#else
    std::cout<<"max(|numerical-exact|)                = "<<max_error          <<std::endl;
    std::cout<<"max(|numerical-exact|) / max(|exact|) = "<<max_error/max_exact<<std::endl;
#endif

#endif
  }

};

#endif
