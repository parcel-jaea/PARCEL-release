// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef CONVDIFF_HPP_
#define CONVDIFF_HPP_

// 3D 7-point convection-diffusion (非対称行列, ks_arnoldi 動作確認用)
//   A = -∇² + 2c·∇  (Dirichlet BC on cube)
//   各方向 1D tridiag: A[i, i-1] = +(1+c), A[i, i] = -2, A[i, i+1] = +(1-c)
//   3D 7-point stencil:
//     diagonal       = -6
//     left/south/bot = +(1+c)   (sub-diagonal)
//     right/north/top= +(1-c)   (super-diagonal)
//
//   解析固有値 (1D tridiag Toeplitz 公式):
//     λ_k(1D) = -2 + 2·sqrt((1+c)(1-c))·cos(kπ/(N+1))
//             = -2 + 2·sqrt(1-c²)·cos(kπ/(N+1))
//   3D: λ_{ijk} = -6 + 2·sqrt(1-c²) · [cos(iπ/(N+1)) + cos(jπ/(N+1)) + cos(kπ/(N+1))]
//
//   c による特性 (NX=64 での固有ベクトル基底条件数 κ(V) ≈ ((1+c)/(1-c))^N):
//     c=0.05  : κ(V) ≈ 770       (textbook 標準, 全実固有値, 機械精度収束想定)
//     c=0.1   : κ(V) ≈ 7e5       (収束やや困難)
//     c=0.5   : κ(V) ≈ 1e30      (高度に非対称, eigvec 基底悪条件)
//     c=2     : 1-c² < 0 → 複素共役対固有値, pseudo-spectrum 顕著 (教材)

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
// 注: crs_mat<T>, PARCEL_INT 等は事前に poisson_cpu.hpp 経由で include 済前提

#ifndef CONVDIFF_C
#define CONVDIFF_C 0.05      // default: low Peclet number (textbook 標準テスト)
#endif

template<class T>
class convdiff_equation {

  double c_ = CONVDIFF_C;   // convection amplitude (-DCONVDIFF_C=<val> で上書き可)

public :

  void   set_c(double c) { c_ = c; }
  double get_c() const   { return c_; }


  PARCEL_INT get_nnz(PARCEL_INT n){
    // 7-pt 構造 (中心は値 0 だが疎構造には含めて Poisson と同形)
    PARCEL_INT nnz_ = 0;
    PARCEL_INT nx = n, ny = n, nz = n;
    for(auto iz = 0; iz < nz; iz++)
      for(auto iy = 0; iy < ny; iy++)
        for(auto ix = 0; ix < nx; ix++){
          PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
          PARCEL_INT icw = ix==0      ? -1 : icc - 1;
          PARCEL_INT ice = ix==(nx-1) ? -1 : icc + 1;
          PARCEL_INT ics = iy==0      ? -1 : icc - nx;
          PARCEL_INT icn = iy==(ny-1) ? -1 : icc + nx;
          PARCEL_INT icb = iz==0      ? -1 : icc - nx*ny;
          PARCEL_INT ict = iz==(nz-1) ? -1 : icc + nx*ny;
          if(0<=icb) nnz_++;
          if(0<=ics) nnz_++;
          if(0<=icw) nnz_++;
          if(0<=icc) nnz_++;   // center (= 0)
          if(0<=ice) nnz_++;
          if(0<=icn) nnz_++;
          if(0<=ict) nnz_++;
        }
    return nnz_;
  }

  // A = -∇² + 2c·∇  convection-diffusion 7-pt
  //   sub-diag (-x, -y, -z): +(1+c)
  //   diag (center)          : -6
  //   sup-diag (+x, +y, +z): +(1-c)
  void make_matrix(PARCEL_INT n, crs_mat<T> &crsA){
    PARCEL_INT nnz = 0;
    PARCEL_INT nx = n, ny = n, nz = n;
    T cb = 1.0 + c_;   // -z (sub)
    T cs = 1.0 + c_;   // -y
    T cw = 1.0 + c_;   // -x
    T cc = -6.0;       // diag
    T ce = 1.0 - c_;   // +x (sup)
    T cn = 1.0 - c_;   // +y
    T ct = 1.0 - c_;   // +z
    crsA.setValue_crsRow_ptr(0,0);
    for(auto iz = 0; iz < nz; iz++)
      for(auto iy = 0; iy < ny; iy++)
        for(auto ix = 0; ix < nx; ix++){
          PARCEL_INT icc = ix + iy*nx + iz*nx*ny;
          PARCEL_INT icw = ix==0      ? -1 : icc - 1;
          PARCEL_INT ice = ix==(nx-1) ? -1 : icc + 1;
          PARCEL_INT ics = iy==0      ? -1 : icc - nx;
          PARCEL_INT icn = iy==(ny-1) ? -1 : icc + nx;
          PARCEL_INT icb = iz==0      ? -1 : icc - nx*ny;
          PARCEL_INT ict = iz==(nz-1) ? -1 : icc + nx*ny;
          crsA.setValue_crsRow_ptr(nnz,icc);
          if(0<=icb){ crsA.setValue_matA(cb,nnz); crsA.setValue_crsCol(icb,nnz); nnz++; }
          if(0<=ics){ crsA.setValue_matA(cs,nnz); crsA.setValue_crsCol(ics,nnz); nnz++; }
          if(0<=icw){ crsA.setValue_matA(cw,nnz); crsA.setValue_crsCol(icw,nnz); nnz++; }
          if(0<=icc){ crsA.setValue_matA(cc,nnz); crsA.setValue_crsCol(icc,nnz); nnz++;
                      crsA.setValue_matDiag(cc,icc); }
          if(0<=ice){ crsA.setValue_matA(ce,nnz); crsA.setValue_crsCol(ice,nnz); nnz++; }
          if(0<=icn){ crsA.setValue_matA(cn,nnz); crsA.setValue_crsCol(icn,nnz); nnz++; }
          if(0<=ict){ crsA.setValue_matA(ct,nnz); crsA.setValue_crsCol(ict,nnz); nnz++; }
        }
    crsA.setValue_crsRow_ptr(nnz,nx*ny*nz);
  }

  // 線形ソルバー用: convdiff は manufactured solution 提供せず (固有値テスト専用)
  void make_right_vector(PARCEL_INT n, std::vector<T> &b){
    b.assign((size_t)n*n*n, T(1.0));
  }

  void check_solution(std::vector<T> &x, PARCEL_INT n){
    std::cout << "[convdiff] check_solution: 製造解なし (固有値専用テスト行列)" << std::endl;
  }

  // Convection-diffusion 解析固有値 (|λ| 降順)
  //   λ_{ijk} = -6 + 2·sqrt(1-c²) · [cos(iπ/(N+1)) + cos(jπ/(N+1)) + cos(kπ/(N+1))]
  //   |c| < 1: sqrt 実数 → 全実固有値, Im(λ)=0
  //   |c| > 1: sqrt 純虚 → 複素共役対, Re(λ)=-6
  void top_eigenvalues_complex(PARCEL_INT n, int k,
                                std::vector<double> &re_out,
                                std::vector<double> &im_out){
    const double pi = std::acos(-1.0);
    double disc = 1.0 - c_*c_;
    double sqrt_re, sqrt_im;
    if(disc >= 0.0){ sqrt_re = std::sqrt( disc); sqrt_im = 0.0;              }
    else           { sqrt_re = 0.0;              sqrt_im = std::sqrt(-disc); }
    std::vector<double> cs(n);
    for(PARCEL_INT i = 0; i < n; i++) cs[i] = std::cos((double)(i+1)*pi/(double)(n+1));
    std::vector<std::pair<double,double> > all;
    all.reserve((size_t)n*n*n);
    for(PARCEL_INT iz = 0; iz < n; iz++)
      for(PARCEL_INT iy = 0; iy < n; iy++)
        for(PARCEL_INT ix = 0; ix < n; ix++){
          double s  = cs[ix] + cs[iy] + cs[iz];
          double lr = -6.0 + 2.0 * sqrt_re * s;
          double li =        2.0 * sqrt_im * s;
          all.emplace_back(lr, li);
        }
    std::sort(all.begin(), all.end(),
              [](const std::pair<double,double>& a, const std::pair<double,double>& b){
                double ma = a.first*a.first + a.second*a.second;
                double mb = b.first*b.first + b.second*b.second;
                return ma > mb;
              });
    size_t kk = std::min((size_t)k, all.size());
    re_out.resize(kk); im_out.resize(kk);
    for(size_t i = 0; i < kk; i++){ re_out[i] = all[i].first; im_out[i] = all[i].second; }
  }

  // |λ| 昇順 (= 0 に最も近い側から) k 個。逆べき乗法の検証に使う。
  void bottom_eigenvalues_complex(PARCEL_INT n, int k,
                                  std::vector<double> &re_out,
                                  std::vector<double> &im_out){
    top_eigenvalues_complex(n, (int)((size_t)n*n*n), re_out, im_out);
    std::reverse(re_out.begin(), re_out.end());
    std::reverse(im_out.begin(), im_out.end());
    size_t kk = std::min((size_t)k, re_out.size());
    re_out.resize(kk); im_out.resize(kk);
  }
};

#endif
