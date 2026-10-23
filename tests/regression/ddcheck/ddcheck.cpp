// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
// ddcheck: double-double (dd) 演算が黙って double に落ちていないかを確かめる回帰試験
//
// 背景
//   -ffast-math を付けてビルドすると ddlib の dd 加算の誤差項 (TwoSum) が最適化で消え、
//   lo が常に 0、つまり dd == double になる。
//   回帰試験の calcDD / dataDD は解の誤差を double 水準の閾値 (1e-6〜1e-10) で見ているので、
//   dd が double に落ちてもそのまま OK になる。ライブラリは __FAST_MATH__ を #error で拒否するが
//   (PARCEL_ALLOW_FAST_MATH で解除可)、デバイス側の dd カーネル (mixed_precision_dd*.hpp,
//   ddlib_macro.hpp のマクロ) はその保険以外に守りがない。
//
// やること
//   ソルバが dd 構成で実際に通るのと同じ parcel::blas の経路 (dot / axpy / gemv / SpMV /
//   calc_res) に、「厳密な答えが既知で、double では表せない (または double で計算すると
//   0 に潰れる)」入力を与え、結果が dd の精度で正しいかを見る。
//     calcDD : データ double, 演算 dd  (parcel_krylov_sample_verify.cpp の PARCEL_RUN と同じ型)
//     dataDD : データ dd,     演算 dd
//   入力はすべて 2 の冪の和で作ってあり、dd で正しく計算されれば誤差は厳密に 0 になる。
//   double に落ちると相対誤差は O(1) (打ち消し型) か 1e-17 程度 (端数型) になり、
//   閾値 DDCHECK_TOL (既定 1e-28) との間に 11 桁以上の差がある。
//
// 出力 (rank 0。標準出力と DDCHECK_OUTPUT のファイル (既定 ddcheck_result.txt) の両方)
//   [DDCHECK] <名前> err=<相対誤差> dd_ok=YES|NO  ...補足
//   [DDCHECK] RESULT PASS|FAIL
//   FAIL なら終了コード 1。
//
// 実行時オプション (環境変数)
//   DDCHECK_N       大域ベクトル長 (既定 32768。512 の倍数、4096 以上)
//   DDCHECK_OUTPUT  結果ファイル。none で書かない
#include "../../../Include/parcel_sycl.hpp"
#include "../../verify/parcel_default.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef DDCHECK_TOL
#define DDCHECK_TOL 1.0e-28
#endif

using DD = parcel_doubledouble;

// ---------------------------------------------------------------- 判定側の算術
//   判定そのものが -ffast-math で崩れないよう、この範囲は厳密な浮動小数点の意味で計算させる
//   (陰性対照を -ffast-math でビルドするため)。NaN / Inf はビット列で調べる
//   (-ffinite-math-only では std::isfinite が消されうる)。
#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif
static bool finite_bits(double v){
  uint64_t u; std::memcpy(&u, &v, sizeof(u));
  return ((u >> 52) & 0x7ff) != 0x7ff;
}
// (g_hi + g_lo) と厳密値 (e_hi + e_lo) の差を |scale| で割る。
//   入力は厳密値に近いことを前提に、hi どうし・lo どうしを先に引く (g_hi - e_hi は
//   ほぼ厳密)。非有限は 1e300 を返す (必ず NO になる)。
static double dd_err(double g_hi, double g_lo, double e_hi, double e_lo, double scale){
  if(!finite_bits(g_hi) || !finite_bits(g_lo)) return 1.0e300;
  const double d = (g_hi - e_hi) + (g_lo - e_lo);
  const double r = std::fabs(d) / std::fabs(scale);
  return finite_bits(r) ? r : 1.0e300;
}
static double pow2(int e){ return std::ldexp(1.0, e); }
#if defined(__clang__)
#pragma float_control(pop)
#endif

// ---------------------------------------------------------------- 出力と集計
struct reporter {
  int rank = 0;
  int n_check = 0, n_fail = 0, n_info = 0;
  std::ostringstream buf;
  void line(const std::string &s){
    if(rank == 0){ std::cout << s << std::endl; buf << s << "\n"; }
  }
  // counted=false は既知の問題の記録用 (合否に数えない)
  void check(const std::string &name, double err, double dbl_err, const std::string &extra,
             bool counted = true){
    const bool ok = finite_bits(err) && err <= DDCHECK_TOL;
    std::ostringstream s;
    s << "[DDCHECK] " << name << " err=" << std::scientific << std::setprecision(3) << err
      << " dd_ok=" << (ok ? "YES" : "NO")
      << "  (double would give " << std::setprecision(2) << dbl_err << ")";
    if(!extra.empty()) s << "  " << extra;
    if(!counted) s << "  [KNOWN_ISSUE: not counted]";
    line(s.str());
    if(counted){ n_check++; if(!ok) n_fail++; } else { n_info++; }
  }
};

static std::string hl(double hi, double lo){
  std::ostringstream s;
  s << std::scientific << std::setprecision(17) << hi << " + " << std::setprecision(6) << lo;
  return s.str();
}

// 全ランクの最大値 (誤差の集約)
//   MPI なしのビルド (-DPARCEL_MPI=0) では 1 プロセスなのでそのまま
static double allmax(double v){
  double r = 0.0;
#if PARCEL_MPI
  MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
  r = v;
#endif
  return r;
}

// ---------------------------------------------------------------- ベクトルの入出力
template<class TD>
static constexpr bool is_dd(){ return std::is_same<TD, DD>::value; }

// hi / lo をデバイスのベクトルへ。double データのときは lo が 0 であること (呼び出し側の責任)
template<class TD>
static void put(parcel::vector<TD> &v, std::vector<double> &hi, std::vector<double> &lo){
  if constexpr (is_dd<TD>()){ v.setValue(hi.data(), lo.data()); }
  else                      { v.setValue(hi.data()); }
}
template<class TD>
static void get(parcel::vector<TD> &v, std::vector<double> &hi, std::vector<double> &lo){
  hi.assign(v.size(), 0.0); lo.assign(v.size(), 0.0);
  if constexpr (is_dd<TD>()){ v.getValue_hi(hi.data()); v.getValue_lo(lo.data()); }
  else                      { v.getValue(hi.data()); }
}
static void put_scalar(parcel::vector<DD> &v, double hi, double lo){
  v.setValue(&hi, &lo);
}

// ---------------------------------------------------------------- 問題の配置
struct layout {
  PARCEL_INT N = 32768;         // 大域長
  PARCEL_INT nstart = 0, nend = 0, n = 0;
  int rank = 0, nproc = 1;
};

// 端数型の要素位置 (1 + M*t の t を置く) と、打ち消し型の組の位置
static bool at_tail(PARCEL_INT g){ return g % 512 == 5; }
static bool at_pairA(PARCEL_INT g){ return g % 512 == 7; }
static bool at_pairB(PARCEL_INT g){ return g % 512 == 9; }
// 要素ごとの尺度 2^(g%8) (全要素同じ値だと添字の取り違えを見逃すため)
static double scale_of(PARCEL_INT g){ return pow2((int)(g % 8)); }

// ---------------------------------------------------------------- 本体
//   TD = データ精度 (parcel_double なら calcDD、parcel_doubledouble なら dataDD の経路)。
//   演算精度 (Tdot / Taxpy / Tspmv) は常に dd。
template<class TD>
static void run_suite(const char *label, sycl::queue &q, parcel::comm &comm,
                      parcel::blas<PARCEL_MAX_FLOTING_TYPE> &blas, const layout &L,
                      reporter &R){
  const double t   = pow2(-60);
  const PARCEL_INT n = L.n;
  std::vector<double> hi(n), lo(n), gh, gl;
  const std::string pre = std::string(label) + ".";

  // 大域の個数 (厳密値の計算に使う)
  PARCEL_INT M_tail = 0, M_pair = 0;
  for(PARCEL_INT g = 0; g < L.N; g++){ if(at_tail(g)) M_tail++; if(at_pairA(g)) M_pair++; }

  parcel::vector<DD> one(1, &q), zero(1, &q), alpha(1, &q), beta(1, &q);
  put_scalar(one, 1.0, 0.0);
  put_scalar(zero, 0.0, 0.0);

  parcel::vector<TD> x(n, &q), y(n, &q), z(n, &q);

  // ---- dot 1: 端数型。 x = (1, t, t, ...) , y = 1  ->  1 + M t   (M t < 2^-53)
  //   double の和は並べ方によらず 1 に丸まる (t の部分和は M t < 半 ulp)。
  //   TwoSum の誤差項 (-ffast-math で消えるもの) がそのまま効く。
  {
    for(PARCEL_INT i = 0; i < n; i++){
      const PARCEL_INT g = L.nstart + i;
      hi[i] = (g == 0) ? 1.0 : (at_tail(g) ? t : 0.0); lo[i] = 0.0;
    }
    put(x, hi, lo);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 1.0; lo[i] = 0.0; }
    put(y, hi, lo);
    blas.template dot<TD, TD, DD>(x, y);
    const ddlib::ddtype r = blas.get_dd();
    const double e_lo = (double)M_tail * t;
    const double err  = dd_err(r.get_hi(), r.get_lo(), 1.0, e_lo, 1.0 + e_lo);
    const double tail = dd_err(r.get_hi(), r.get_lo(), 1.0, e_lo, e_lo);
    std::ostringstream s;
    s << "got=" << hl(r.get_hi(), r.get_lo()) << " exact=1 + " << (double)M_tail << "*2^-60"
      << " tail_err=" << std::scientific << std::setprecision(3) << tail;
    R.check(pre + "dot_tail", err, e_lo / (1.0 + e_lo), s.str());
  }

  // ---- dot 2: 積の誤差項の打ち消し。 a = 1 + 2^-30
  //   (x,y) = (a, a) と (-(1 + 2^-29), 1) の組 -> a^2 - (1 + 2^-29) = 2^-60 が M 組
  //   double の積は 2^-60 を落とし、残りは並べ方によらず厳密に 0 に打ち消し合う。
  {
    const double a = 1.0 + pow2(-30), b = 1.0 + pow2(-29);
    std::vector<double> yh(n), yl(n, 0.0);
    for(PARCEL_INT i = 0; i < n; i++){
      const PARCEL_INT g = L.nstart + i;
      hi[i] = at_pairA(g) ? a : (at_pairB(g) ? -b : 0.0); lo[i] = 0.0;
      yh[i] = at_pairA(g) ? a : (at_pairB(g) ? 1.0 : 0.0);
    }
    put(x, hi, lo); put(y, yh, yl);
    blas.template dot<TD, TD, DD>(x, y);
    const ddlib::ddtype r = blas.get_dd();
    const double e = (double)M_pair * t;
    const double err = dd_err(r.get_hi(), r.get_lo(), e, 0.0, e);
    R.check(pre + "dot_prod_cancel", err, 1.0,
            "got=" + hl(r.get_hi(), r.get_lo()) + " exact=M*2^-60");
  }

  // ---- dot 3 (dd データのみ): lo の打ち消し。 x = (1, t) と (-1, 0) の組, y = 1 -> M t
  if constexpr (is_dd<TD>()){
    for(PARCEL_INT i = 0; i < n; i++){
      const PARCEL_INT g = L.nstart + i;
      hi[i] = at_pairA(g) ? 1.0 : (at_pairB(g) ? -1.0 : 0.0);
      lo[i] = at_pairA(g) ? t : 0.0;
    }
    put(x, hi, lo);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 1.0; lo[i] = 0.0; }
    put(y, hi, lo);
    blas.template dot<TD, TD, DD>(x, y);
    const ddlib::ddtype r = blas.get_dd();
    const double e = (double)M_pair * t;
    const double err = dd_err(r.get_hi(), r.get_lo(), e, 0.0, e);
    R.check(pre + "dot_lo_cancel", err, 1.0,
            "got=" + hl(r.get_hi(), r.get_lo()) + " exact=M*2^-60");
  }

  // ---- axpy: y = alpha x + beta y,  alpha = 1 + 2^-60 (dd), beta = 1,
  //   x_g = s_g = 2^(g%8), y_g = -s_g  ->  y_g = s_g 2^-60 (double で表せる)
  //   double の alpha は 1 に丸まり結果は 0 になる。
  {
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = scale_of(L.nstart + i); lo[i] = 0.0; }
    put(x, hi, lo);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = -scale_of(L.nstart + i); }
    put(y, hi, lo);
    put_scalar(alpha, 1.0, t); put_scalar(beta, 1.0, 0.0);
    blas.template axpy<DD, DD, TD, TD, DD>(alpha, beta, x, y);
    blas.wait();
    get(y, gh, gl);
    double err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    R.check(pre + "axpy", allmax(err), 1.0, "y = (1+2^-60) x - x, max over elements");
  }

  // ---- axpy (3 引数): z = alpha x + beta y,  alpha = 1 + 2^-60, beta = -1, y = x  ->  z = s_g 2^-60
  {
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = scale_of(L.nstart + i); lo[i] = 0.0; }
    put(x, hi, lo); put(y, hi, lo);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 0.0; }
    put(z, hi, lo);
    put_scalar(alpha, 1.0, t); put_scalar(beta, -1.0, 0.0);
    blas.template axpy<DD, DD, TD, TD, TD, DD>(alpha, beta, x, y, z);
    blas.wait();
    get(z, gh, gl);
    double err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    R.check(pre + "axpyz", allmax(err), 1.0, "z = (1+2^-60) x - x, max over elements");
  }

  // ---- axpy 端数型 (dd データのみ): y = x + y, x = 1, y = s_g 2^-60 (g%7)  ->  1 + s_g 2^-60
  //   和の丸め誤差 (TwoSum) が dd の lo に残るかを見る。
  if constexpr (is_dd<TD>()){
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 1.0; lo[i] = 0.0; }
    put(x, hi, lo);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = pow2((int)((L.nstart + i) % 7)) * t; }
    put(y, hi, lo);
    put_scalar(alpha, 1.0, 0.0); put_scalar(beta, 1.0, 0.0);
    blas.template axpy<DD, DD, TD, TD, DD>(alpha, beta, x, y);
    blas.wait();
    get(y, gh, gl);
    double err = 0.0, tail = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double s = pow2((int)((L.nstart + i) % 7)) * t;
      err  = std::max(err,  dd_err(gh[i], gl[i], 1.0, s, 1.0 + s));
      tail = std::max(tail, dd_err(gh[i], gl[i], 1.0, s, s));
    }
    std::ostringstream s;
    s << "y = x + y -> 1 + 2^-60..2^-54, tail_err=" << std::scientific << std::setprecision(3)
      << allmax(tail);
    R.check(pre + "axpy_tail", allmax(err), pow2(-54), s.str());
  }

  // ---- gemv (pcg の pstep 分まとめた x 更新と同じ呼び方): y = one * A c + one * y
  //   A は n x 3 (列優先, ld = n)、列 = (s_g, s_g t, -s_g)、c = (1, 1, 1)、y = 0
  //   -> y_g = s_g t。行内の和 s + s t - s は dd の TwoSum が要る。
  {
    const PARCEL_INT m = 3;
    parcel::vector<TD> A((size_t)n * m, &q);
    std::vector<double> ah((size_t)n * m), al((size_t)n * m, 0.0);
    for(PARCEL_INT i = 0; i < n; i++){
      const double s = scale_of(L.nstart + i);
      ah[i] = s; ah[i + n] = s * t; ah[i + 2 * (size_t)n] = -s;
    }
    put(A, ah, al);
    parcel::vector<DD> c(m, &q);
    std::vector<double> ch = {1.0, 1.0, 1.0}, cl = {0.0, 0.0, 0.0};
    c.setValue(ch.data(), cl.data());
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 0.0; lo[i] = 0.0; }
    put(y, hi, lo);
    blas.template gemv<DD, DD, TD, DD, TD, DD>(m, one, one, A, c, y, n);
    blas.wait();
    get(y, gh, gl);
    double err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    R.check(pre + "gemv_accum", allmax(err), 1.0, "y = [s, s*2^-60, -s] . (1,1,1)");

    // 係数 c の lo が使われるか (既定では記録のみで合否に数えない)。
    //   parcel_blas.hpp の ddgemv_main で x (= 係数ベクトル) が dd の経路。pcg (pstep > 1) の
    //   x 更新と pgmres の直交化の gemv が同じ型の組で呼ぶ。
    //   DDCHECK_STRICT_GEMV=1 で合否に数える。
    for(PARCEL_INT i = 0; i < n; i++){
      const double s = scale_of(L.nstart + i);
      ah[i] = s; ah[i + n] = s; ah[i + 2 * (size_t)n] = 0.0;
    }
    put(A, ah, al);
    ch = {1.0, -1.0, 0.0}; cl = {t, 0.0, 0.0};
    c.setValue(ch.data(), cl.data());
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 0.0; lo[i] = 0.0; }
    put(y, hi, lo);
    blas.template gemv<DD, DD, TD, DD, TD, DD>(m, one, one, A, c, y, n);
    blas.wait();
    get(y, gh, gl);
    err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    bool strict = false;
    if(const char *e = std::getenv("DDCHECK_STRICT_GEMV")){ strict = (std::atoi(e) != 0); }
    R.check(pre + "gemv_coef_lo", allmax(err), 1.0,
            "y = [s, s] . (1+2^-60, -1)", strict);
  }

  // ---- SpMV / calc_res (CRS, MPI 分割、袖通信あり)
  //   行 g の非零は列 c0, c0+1, c0+2 (c0 = min(g, N-3))、値 s_g * (1, 2^-60, -1)
  //   x = 1 -> (A x)_g = s_g 2^-60。列が隣のランクにまたがるので袖通信も通る。
  {
    std::vector<PARCEL_INT> rp(n + 1), cl;
    std::vector<double> av;
    rp[0] = 0;
    for(PARCEL_INT i = 0; i < n; i++){
      const PARCEL_INT g  = L.nstart + i;
      const PARCEL_INT c0 = std::min<PARCEL_INT>(g, L.N - 3);
      const double s = scale_of(g);
      cl.push_back(c0);     av.push_back(s);
      cl.push_back(c0 + 1); av.push_back(s * t);
      cl.push_back(c0 + 2); av.push_back(-s);
      rp[i + 1] = rp[i] + 3;
    }
    parcel::matrix<TD> Mat(&q);
    Mat.template set_crsMat<double>(&comm, L.nstart, L.nend, (size_t)n, (size_t)rp[n],
                                    rp.data(), cl.data(), av.data());
    const PARCEL_INT nadd = Mat.get_recv_size();
    parcel::vector<TD> xs((size_t)n + nadd, &q), ys(n, &q), bs(n, &q), rs(n, &q);
    std::vector<double> xh((size_t)n + nadd, 1.0), xl((size_t)n + nadd, 0.0);
    put(xs, xh, xl);
    for(PARCEL_INT i = 0; i < n; i++){ hi[i] = 0.0; lo[i] = 0.0; }
    put(ys, hi, lo); put(bs, hi, lo); put(rs, hi, lo);

    blas.template SpMV<DD, DD, TD, TD, TD, DD>(one, zero, Mat, xs, ys);
    blas.wait();
    get(ys, gh, gl);
    double err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    R.check(pre + "spmv", allmax(err), 1.0, "row = s*(1, 2^-60, -1), x = 1");

    // r = b - A x (b = 0) -> -s_g 2^-60。pcg の真の残差と同じ呼び方。
    blas.template calc_res<TD, TD, TD, TD, DD>(Mat, xs, bs, rs);
    blas.wait();
    get(rs, gh, gl);
    err = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      const double e = -scale_of(L.nstart + i) * t;
      err = std::max(err, dd_err(gh[i], gl[i], e, 0.0, e));
    }
    R.check(pre + "calc_res", allmax(err), 1.0, "r = 0 - A x");
  }
}

// ホスト側の ddlib::ddtype (MPI の dd 集約もこれを使う)
static void run_host(reporter &R){
  const double t = pow2(-60);
  {
    ddlib::ddtype a(1.0, 0.0), b(t, 0.0);
    ddlib::ddtype c = a + b;          // 1 + 2^-60
    ddlib::ddtype d = c - a;          // 2^-60
    const double e1 = dd_err(c.get_hi(), c.get_lo(), 1.0, t, 1.0 + t);
    const double e2 = dd_err(d.get_hi(), d.get_lo(), t, 0.0, t);
    R.check("host.add", std::max(e1, e2), 1.0,
            "(1 + 2^-60) - 1 = " + hl(d.get_hi(), d.get_lo()));
  }
  {
    const double a = 1.0 + pow2(-30), b = 1.0 + pow2(-29);
    ddlib::ddtype p(a, 0.0), q(b, 0.0);
    ddlib::ddtype r = p * p - q;      // 2^-60
    const double e = dd_err(r.get_hi(), r.get_lo(), t, 0.0, t);
    R.check("host.mul", e, 1.0, "(1+2^-30)^2 - (1+2^-29) = " + hl(r.get_hi(), r.get_lo()));
  }
}

int main(int argc, char *argv[]){
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
#else
  (void)argc; (void)argv;
#endif
  int rc = 0;
  {
    reporter R;
    layout L;
#if PARCEL_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &L.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &L.nproc);
#else
    L.rank = 0; L.nproc = 1;
#endif
    R.rank = L.rank;
    if(const char *e = std::getenv("DDCHECK_N")){
      const long v = std::atol(e);
      if(v >= 4096 && v % 512 == 0){ L.N = (PARCEL_INT)v; }
    }
    const PARCEL_INT nl = L.N / L.nproc;
    L.nstart = nl * L.rank;
    L.nend   = (L.rank == L.nproc - 1) ? L.N - 1 : nl * (L.rank + 1) - 1;
    L.n      = L.nend - L.nstart + 1;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
#if PARCEL_MPI
    parcel::comm comm(MPI_COMM_WORLD, &q);
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> blas(&q, MPI_COMM_WORLD);
#else
    parcel::comm comm(&q);
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> blas(&q);
#endif

    {
      std::ostringstream s;
      s << "[DDCHECK] device=" << q.get_device().get_info<sycl::info::device::name>()
        << " np=" << L.nproc << " N=" << L.N << " tol=" << std::scientific
        << std::setprecision(1) << (double)DDCHECK_TOL
#if defined(__FAST_MATH__)
        << " fast_math=ON"
#else
        << " fast_math=off"
#endif
        << " build_calc_type=" << (std::is_same<parcel_solver_calc_type, DD>::value ? "dd" : "not-dd");
      R.line(s.str());
    }

    run_host(R);
    run_suite<parcel_double>("calcDD", q, comm, blas, L, R);
    run_suite<parcel_doubledouble>("dataDD", q, comm, blas, L, R);

    const bool pass = (R.n_fail == 0 && R.n_check > 0);
    {
      std::ostringstream s;
      s << "[DDCHECK] checks=" << R.n_check << " failed=" << R.n_fail
        << " known_issues=" << R.n_info;
      R.line(s.str());
    }
    R.line(std::string("[DDCHECK] RESULT ") + (pass ? "PASS" : "FAIL"));
    rc = pass ? 0 : 1;

    // rank 0 が自分でファイルにも書く (MPI 実装によっては標準出力がジョブログへ行くため)
    if(L.rank == 0){
      std::string path = "ddcheck_result.txt";
      if(const char *e = std::getenv("DDCHECK_OUTPUT")){ path = e; }
      if(path != "none"){
        std::ofstream f(path);
        f << R.buf.str();
      }
    }
  }
#if PARCEL_MPI
  MPI_Finalize();
#endif
  return rc;
}
