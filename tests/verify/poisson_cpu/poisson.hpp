// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef POISSON_HPP_
#define POISSON_HPP_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include "mat.hpp"

template<class T>
class poisson_equation{

public :

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
	  if(max_error < error){
	    max_error = error;
	    max_exact = exact;
	  }
	}
      }
    }


    std::cout<<"max(|x_numerical - x_exact|)                  = "<<max_error          <<std::endl;
    std::cout<<"max(|x_numerical - x_exact|) / max(|x_exact|) = "<<max_error/max_exact<<std::endl;

  }

  // 3D 7-pt Poisson (Dirichlet BC, A = -∂²/∂x² - ...) の解析固有値
  //   λ_{ijk} = -6 + 2cos(iπ/(N+1)) + 2cos(jπ/(N+1)) + 2cos(kπ/(N+1)),  i,j,k ∈ [1,N]
  //   |λ| 降順 (= 最も負の側から) k 個を返す。返す値は実数 λ そのもの (負)。
  void top_eigenvalues(PARCEL_INT n, int k, std::vector<double> &out){
    const double pi = std::acos(-1.0);
    std::vector<double> c(n);
    for(PARCEL_INT i = 0; i < n; i++) c[i] = std::cos((double)(i+1)*pi/(double)(n+1));
    std::vector<double> all;
    all.reserve((size_t)n*(size_t)n*(size_t)n);
    for(PARCEL_INT iz = 0; iz < n; iz++)
      for(PARCEL_INT iy = 0; iy < n; iy++)
        for(PARCEL_INT ix = 0; ix < n; ix++)
          all.push_back(-6.0 + 2.0*(c[ix] + c[iy] + c[iz]));
    std::sort(all.begin(), all.end(),
              [](double a, double b){ return std::fabs(a) > std::fabs(b); });
    out.assign(all.begin(), all.begin() + std::min((size_t)k, all.size()));
  }

  // |λ| 昇順 (= 0 に最も近い側から) k 個。逆べき乗法の検証に使う。
  void bottom_eigenvalues(PARCEL_INT n, int k, std::vector<double> &out){
    const double pi = std::acos(-1.0);
    std::vector<double> c(n);
    for(PARCEL_INT i = 0; i < n; i++) c[i] = std::cos((double)(i+1)*pi/(double)(n+1));
    std::vector<double> all;
    all.reserve((size_t)n*(size_t)n*(size_t)n);
    for(PARCEL_INT iz = 0; iz < n; iz++)
      for(PARCEL_INT iy = 0; iy < n; iy++)
        for(PARCEL_INT ix = 0; ix < n; ix++)
          all.push_back(-6.0 + 2.0*(c[ix] + c[iy] + c[iz]));
    std::sort(all.begin(), all.end(),
              [](double a, double b){ return std::fabs(a) < std::fabs(b); });
    out.assign(all.begin(), all.begin() + std::min((size_t)k, all.size()));
  }

  void bottom_eigenvalues_complex(PARCEL_INT n, int k,
                                  std::vector<double> &re_out,
                                  std::vector<double> &im_out){
    bottom_eigenvalues(n, k, re_out);
    im_out.assign(re_out.size(), 0.0);
  }

  // 複素固有値インタフェース (convdiff と統一; Poisson は実対称なので im=0)
  void top_eigenvalues_complex(PARCEL_INT n, int k,
                                std::vector<double> &re_out,
                                std::vector<double> &im_out){
    top_eigenvalues(n, k, re_out);
    im_out.assign(re_out.size(), 0.0);
  }

};

#endif
