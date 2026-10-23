// =========================================================================
// parcel_version.hpp
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// PARCEL_SYCL の版。版の値はこのファイルの PARCEL_VERSION_MAJOR / MINOR / PATCH の 3 つだけが正で、
// 他はここから取る (C / Fortran の API の parcel_get_version()・parcel_version_string() はラッパが
// この値を返し、Python の parcel.library_version() はネイティブ層がこの値を返す)。
// ここから取れないもの (Python のパッケージの版 python/pyproject.toml・parcel.__version__、
// spack のレシピの version()、CITATION.cff の version) は tests/check_version.py で照合する。
//
//   PARCEL_VERSION_MAJOR / _MINOR / _PATCH   版の番号 (整数)
//   PARCEL_VERSION_STRING                   "MAJOR.MINOR.PATCH" (文字列リテラル)
//   PARCEL_VERSION_NUMBER                   MAJOR * 10000 + MINOR * 100 + PATCH (1.2.0 なら 10200)
//   PARCEL_VERSION_GE(a, b, c)              この版が a.b.c 以上なら 1 (#if で使える)
//   parcel::version_string()                PARCEL_VERSION_STRING を返す
//   parcel::get_version(&major, &minor, &patch)  番号を返す (nullptr の引数は書かない)
//
// C のプリプロセッサでも読める (C++ の関数は __cplusplus のときだけ)。SYCL・MPI は読まない。
// =========================================================================
#ifndef PARCEL_VERSION_HPP_
#define PARCEL_VERSION_HPP_

#define PARCEL_VERSION_MAJOR 1
#define PARCEL_VERSION_MINOR 2
#define PARCEL_VERSION_PATCH 0

#define PARCEL_VERSION_STR_(x) #x
#define PARCEL_VERSION_XSTR_(x) PARCEL_VERSION_STR_(x)
#define PARCEL_VERSION_STRING                                                     \
  PARCEL_VERSION_XSTR_(PARCEL_VERSION_MAJOR) "." PARCEL_VERSION_XSTR_(PARCEL_VERSION_MINOR) \
  "." PARCEL_VERSION_XSTR_(PARCEL_VERSION_PATCH)
#define PARCEL_VERSION_NUMBER \
  (PARCEL_VERSION_MAJOR * 10000 + PARCEL_VERSION_MINOR * 100 + PARCEL_VERSION_PATCH)
#define PARCEL_VERSION_GE(a, b, c) (PARCEL_VERSION_NUMBER >= ((a) * 10000 + (b) * 100 + (c)))

#ifdef __cplusplus
namespace parcel {

  // 版の文字列 ("1.2.0" など。PARCEL_VERSION_STRING と同じ)
  inline const char *version_string() noexcept { return PARCEL_VERSION_STRING; }

  // 版の番号。nullptr を渡した引数は書かない
  inline void get_version(int *major, int *minor, int *patch) noexcept {
    if(major != nullptr) *major = PARCEL_VERSION_MAJOR;
    if(minor != nullptr) *minor = PARCEL_VERSION_MINOR;
    if(patch != nullptr) *patch = PARCEL_VERSION_PATCH;
  }

} // namespace parcel
#endif

#endif // PARCEL_VERSION_HPP_
