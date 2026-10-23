// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_SOLVER_SELECT_HPP_
#define PARCEL_SOLVER_SELECT_HPP_

// デバッグ用に、ビルドするソルバを絞るためのスイッチ。
//
//   PARCEL は全ソルバがヘッダオンリーで 1 つの翻訳単位にまとまるため、
//   1 精度ぶんのコンパイルに A64FX で 6〜14 分かかる。
//   切り分けで 1 ソルバしか使わないときは、他を外すと待ち時間が縮む。
//
//   使い方 (既定は全ソルバ有効なので、通常のビルドは何も指定しなくてよい):
//     全部外して pgmres だけ:
//       -DPARCEL_ENABLE_ALL_SOLVERS=0 -DPARCEL_ENABLE_PGMRES=1
//     pcg と ks_arnoldi だけ:
//       -DPARCEL_ENABLE_ALL_SOLVERS=0 -DPARCEL_ENABLE_PCG=1 -DPARCEL_ENABLE_KS_ARNOLDI=1
//     ks_arnoldi だけ外す:
//       -DPARCEL_ENABLE_KS_ARNOLDI=0
//
//   **外したソルバを実行時に PARCEL_SOLVER= で指定すると、起動時に
//     名前を出して異常終了する。** 黙って別のソルバを解いて、測ったつもりの
//     ものと違う結果を持ち帰ることがないようにしている。
//
//   回帰試験・性能試験のビルドはここを触らないこと。全ソルバが要る。

#ifndef PARCEL_ENABLE_ALL_SOLVERS
#define PARCEL_ENABLE_ALL_SOLVERS 1
#endif

#ifndef PARCEL_ENABLE_PCG
#define PARCEL_ENABLE_PCG PARCEL_ENABLE_ALL_SOLVERS
#endif
#ifndef PARCEL_ENABLE_PBICGSTAB
#define PARCEL_ENABLE_PBICGSTAB PARCEL_ENABLE_ALL_SOLVERS
#endif
#ifndef PARCEL_ENABLE_PGMRES
#define PARCEL_ENABLE_PGMRES PARCEL_ENABLE_ALL_SOLVERS
#endif
#ifndef PARCEL_ENABLE_PCAGMRES
#define PARCEL_ENABLE_PCAGMRES PARCEL_ENABLE_ALL_SOLVERS
#endif
// Q を陽に作らない CA-GMRES (CAGMRES_Q_IMPLICIT) も PARCEL_ENABLE_PCAGMRES で入る。
//   PARCEL_ENABLE_CHOLQR_PCAGMRES は使えないので、指定されていたら止める (黙って無視しない)。
#ifdef PARCEL_ENABLE_CHOLQR_PCAGMRES
#error "PARCEL_ENABLE_CHOLQR_PCAGMRES は 2026-09-25 に廃止した。cholQR_pcagmres は pcagmres の CAGMRES_Q_IMPLICIT になったので PARCEL_ENABLE_PCAGMRES を使う"
#endif
#ifndef PARCEL_ENABLE_POWER
#define PARCEL_ENABLE_POWER PARCEL_ENABLE_ALL_SOLVERS
#endif
#ifndef PARCEL_ENABLE_INVERSE_POWER
#define PARCEL_ENABLE_INVERSE_POWER PARCEL_ENABLE_ALL_SOLVERS
#endif
#ifndef PARCEL_ENABLE_KS_ARNOLDI
#define PARCEL_ENABLE_KS_ARNOLDI PARCEL_ENABLE_ALL_SOLVERS
#endif

// ---- 依存関係。外したつもりのものが必要になる場合は黙って戻す ----
//   inverse_power は内部で pcg を解く (parcel_krylov_inverse_power.hpp)
#if PARCEL_ENABLE_INVERSE_POWER && !PARCEL_ENABLE_PCG
#undef  PARCEL_ENABLE_PCG
#define PARCEL_ENABLE_PCG 1
#endif
//   inverse_power の A M^-1 モード (set_precon_spectrum) は内部で pgmres を解く
#if PARCEL_ENABLE_INVERSE_POWER && !PARCEL_ENABLE_PGMRES
#undef  PARCEL_ENABLE_PGMRES
#define PARCEL_ENABLE_PGMRES 1
#endif
//   ks_arnoldi のシフト逆反復は内部で pgmres を解く (parcel_krylov_ks.hpp)
#if PARCEL_ENABLE_KS_ARNOLDI && !PARCEL_ENABLE_PGMRES
#undef  PARCEL_ENABLE_PGMRES
#define PARCEL_ENABLE_PGMRES 1
#endif

#if !(PARCEL_ENABLE_PCG || PARCEL_ENABLE_PBICGSTAB || PARCEL_ENABLE_PGMRES || \
      PARCEL_ENABLE_PCAGMRES || PARCEL_ENABLE_POWER || PARCEL_ENABLE_INVERSE_POWER || \
      PARCEL_ENABLE_KS_ARNOLDI)
#error "ソルバが 1 つも有効になっていない (PARCEL_ENABLE_ALL_SOLVERS=0 のまま個別指定を忘れている)"
#endif

#endif
