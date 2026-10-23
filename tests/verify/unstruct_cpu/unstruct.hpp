// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef UNSTRUCT_HPP_
#define UNSTRUCT_HPP_

// 行あたり非零数が一定でない試験行列 (非構造格子を模す)。
//
//   目的:
//     SpMV の実装最適化 (梯子分割 PARCEL_SPMV_LADDER など) は「行あたり非零数が一定」を
//     仮定すると速くなるものがあり、構造格子と非構造格子で効果が逆になりうる。
//     poisson と convdiff は全行 7 非零なので、非構造側の挙動をソルバ単位で確かめるために使う。
//
//   作り方:
//     3 次元 7 点ステンシル (poisson と同じ) を土台に、行ごとに 0〜3 個の追加非零を
//     行番号から決まる疑似乱数で加える。追加分の値は -0.1 で、その分だけ対角の絶対値を
//     増やして対角優位を保つ (反復が発散しないようにするため)。
//     行あたり非零数は内部行で 7〜10、境界行で 4〜10 になる。
//
//   注意:
//     追加分は片側にしか入らないので **行列は非対称**。CG は使えないので、
//     この行列での評価は pgmres / pbicgstab / ks_arnoldi で行う。
//     解析固有値も存在しないため、固有値ソルバの解析値照合は行わない
//     (matrix_selector::has_analytic_eigen() が false を返す)。
//
//   大域生成 (make_matrix) と担当行生成 (matrix_selector::make_matrix_rows) は
//   どちらも build_row() を呼ぶので、同じ行には必ず同じ列・同じ値が入る。
//   検証時の 1 要素ずつの照合 (run_verify) がそのまま通る。

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

template<class T>
class unstruct_equation {
public:
  // 行 icc に足す追加非零の個数 (0〜3)
  static PARCEL_INT nextra(PARCEL_INT icc){
    return (PARCEL_INT)((((std::uint64_t)icc * 2654435761ull) >> 13) % 4ull);
  }
  // t 番目の追加非零の列番号
  static PARCEL_INT extra_col(PARCEL_INT icc, int t, PARCEL_INT ndim){
    std::uint64_t h = (std::uint64_t)icc * 2654435761ull + (std::uint64_t)(t + 1) * 40503ull;
    h ^= h >> 17;
    return (PARCEL_INT)(h % (std::uint64_t)ndim);
  }

  // 行 icc の (列, 値) を列番号の昇順で作る。7 点ステンシル + 追加非零。
  //   1 行あたりの要素は最大 10 個なので固定長配列で組み立て、追加分は挿入で昇順を保つ。
  //   行ごとの確保と整列を避け、大きな NX でも行列生成が遅くならないようにする。
  //   cols / vals は呼び出し側で使い回す前提 (assign は確保済み容量を再利用する)。
  static void build_row(PARCEL_INT n, PARCEL_INT icc,
                        std::vector<PARCEL_INT> &cols, std::vector<T> &vals){
    const PARCEL_INT nx = n, ny = n, nz = n;
    const PARCEL_INT ndim = nx*ny*nz;
    const PARCEL_INT ix = icc % nx;
    const PARCEL_INT iy = (icc / nx) % ny;
    const PARCEL_INT iz = icc / (nx*ny);
    PARCEL_INT c[10];
    T          v[10];
    int k = 0;
    // 7 点ステンシル。この順で詰めると既に昇順になる (poisson と同じ係数・同じ並び)
    if(iz > 0)      { c[k] = icc - nx*ny; v[k] = T(1.0);  k++; }
    if(iy > 0)      { c[k] = icc - nx;    v[k] = T(1.0);  k++; }
    if(ix > 0)      { c[k] = icc - 1;     v[k] = T(1.0);  k++; }
    /* 対角 */        c[k] = icc;         v[k] = T(-6.0); k++;
    if(ix < nx - 1) { c[k] = icc + 1;     v[k] = T(1.0);  k++; }
    if(iy < ny - 1) { c[k] = icc + nx;    v[k] = T(1.0);  k++; }
    if(iz < nz - 1) { c[k] = icc + nx*ny; v[k] = T(1.0);  k++; }
    // 追加非零を昇順を保ったまま挿入する。対角と既存列に重なるものは捨てる
    int nadded = 0;
    const PARCEL_INT ne = nextra(icc);
    for(int t = 0; t < (int)ne; t++){
      const PARCEL_INT jc = extra_col(icc, t, ndim);
      if(jc == icc) continue;
      bool dup = false;
      for(int q = 0; q < k; q++){ if(c[q] == jc){ dup = true; break; } }
      if(dup) continue;
      int pos = k;
      while(pos > 0 && c[pos-1] > jc){ c[pos] = c[pos-1]; v[pos] = v[pos-1]; pos--; }
      c[pos] = jc; v[pos] = T(-0.1);
      k++; nadded++;
    }
    // 対角は非対角の絶対値の合計と同じ大きさにして対角優位を保つ (挿入で位置が動くので探す)
    for(int q = 0; q < k; q++){ if(c[q] == icc){ v[q] = T(-6.0 - 0.1 * (double)nadded); break; } }
    cols.assign(c, c + k);
    vals.assign(v, v + k);
  }

  // 既知解 (poisson と同じ式)
  static T exact_at(PARCEL_INT n, PARCEL_INT icc){
    const double h = 1.0 / (double)(n + 1);
    const PARCEL_INT ix = icc % n;
    const PARCEL_INT iy = (icc / n) % n;
    const PARCEL_INT iz = icc / (n*n);
    const double hx3 = h*h*h;
    return (T)(hx3*hx3 * (double)((ix+1)*(ix+1)) * (double)((iy+1)*(iy+1)) * (double)((iz+1)*(iz+1)));
  }

  // 行を組み立てずに非零数だけ数える。
  //   大きな NX で全行の組み立てを避けるため。
  //   重複の捨て方は build_row() と同じにする (近傍 6 個と、既に採用した追加分を見る)。
  PARCEL_INT get_nnz(PARCEL_INT n){
    const PARCEL_INT nx = n, ny = n, nz = n;
    const PARCEL_INT ndim = nx*ny*nz;
    PARCEL_INT nnz = 0;
    PARCEL_INT nb[9];   // 近傍 6 + 追加 3
    for(PARCEL_INT icc = 0; icc < ndim; icc++){
      const PARCEL_INT ix = icc % nx;
      const PARCEL_INT iy = (icc / nx) % ny;
      const PARCEL_INT iz = icc / (nx*ny);
      int k = 0;
      if(iz > 0)      nb[k++] = icc - nx*ny;
      if(iy > 0)      nb[k++] = icc - nx;
      if(ix > 0)      nb[k++] = icc - 1;
      if(ix < nx - 1) nb[k++] = icc + 1;
      if(iy < ny - 1) nb[k++] = icc + nx;
      if(iz < nz - 1) nb[k++] = icc + nx*ny;
      const PARCEL_INT ne = nextra(icc);
      for(int t = 0; t < (int)ne; t++){
        const PARCEL_INT jc = extra_col(icc, t, ndim);
        if(jc == icc) continue;
        bool dup = false;
        for(int q = 0; q < k; q++){ if(nb[q] == jc){ dup = true; break; } }
        if(dup) continue;
        nb[k++] = jc;
      }
      nnz += (PARCEL_INT)k + 1;   // 近傍 + 採用した追加分 + 対角
    }
    return nnz;
  }

  void make_matrix(PARCEL_INT n, crs_mat<T> &crsA){
    const PARCEL_INT ndim = n*n*n;
    PARCEL_INT nnz = 0;
    std::vector<PARCEL_INT> cols; std::vector<T> vals;
    crsA.setValue_crsRow_ptr(0,0);
    for(PARCEL_INT icc = 0; icc < ndim; icc++){
      build_row(n, icc, cols, vals);
      crsA.setValue_crsRow_ptr(nnz, icc);
      for(size_t j = 0; j < cols.size(); j++){
        crsA.setValue_matA(vals[j], nnz);
        crsA.setValue_crsCol(cols[j], nnz);
        if(cols[j] == icc) crsA.setValue_matDiag(vals[j], icc);
        nnz++;
      }
    }
    crsA.setValue_crsRow_ptr(nnz, ndim);
  }

  // 既知解に対する右辺 b = A x_exact
  void make_right_vector(PARCEL_INT n, std::vector<T> &b){
    const PARCEL_INT ndim = n*n*n;
    b.assign((size_t)ndim, T(0.0));
    std::vector<PARCEL_INT> cols; std::vector<T> vals;
    for(PARCEL_INT icc = 0; icc < ndim; icc++){
      build_row(n, icc, cols, vals);
      double s = 0.0;
      for(size_t j = 0; j < cols.size(); j++) s += (double)vals[j] * (double)exact_at(n, cols[j]);
      b[(size_t)icc] = (T)s;
    }
  }

  void check_solution(std::vector<T> &x, PARCEL_INT n){
    const PARCEL_INT ndim = n*n*n;
    double maxerr = 0.0, maxexact = 0.0;
    const size_t nloc = x.size();
    for(size_t i = 0; i < nloc && (PARCEL_INT)i < ndim; i++){
      const double e = (double)exact_at(n, (PARCEL_INT)i);
      maxerr   = std::max(maxerr, std::fabs((double)x[i] - e));
      maxexact = std::max(maxexact, std::fabs(e));
    }
    std::cout << "=== VERIFY: unstruct solution error vs manufactured exact solution ===\n";
    std::cout << "  max|x - x_exact|                            = " << maxerr << "\n";
    std::cout << "  max|x - x_exact| / max|x_exact|              = "
              << (maxexact > 0.0 ? maxerr/maxexact : 0.0) << std::endl;
  }

  // 解析固有値は存在しない。呼ばれた場合は空を返す (照合は行わない)
  void top_eigenvalues_complex(PARCEL_INT, int, std::vector<double> &re, std::vector<double> &im){
    re.clear(); im.clear();
  }
  void bottom_eigenvalues_complex(PARCEL_INT, int, std::vector<double> &re, std::vector<double> &im){
    re.clear(); im.clear();
  }
};

#endif
