! quick_start_f.f90 : Fortran の API で 1 次元 Laplace 作用素の三重対角行列の連立一次方程式を pcg で解く最小の例
!   MPI あり: mpif90 でコンパイルする。MPI なし: gfortran で examples/sample_f_mpi_stub.f90 (1 プロセス用の mpi モジュール) と組む
! ビルドと実行はマニュアルのクイックスタートを参照
!
! SPDX-License-Identifier: LGPL-3.0-only
! Copyright (c) 日本原子力研究開発機構 システム計算科学センター
program quick_start_f
  use mpi
  use iso_c_binding
  use parcel_sycl_mod
  implicit none
  integer(parcel_int_kind), parameter :: n_global = 1000
  integer(parcel_int_kind) :: nstart, nend, n, nnz, i
  integer(parcel_int_kind), allocatable :: rowptr(:), col(:)
  real(c_double), allocatable :: val(:), b(:), x(:)
  type(parcel_options) :: opt
  type(parcel_result)  :: res
  integer :: myrank, nproc, ierr, rc
  ! 厳密解との比較のしきい値. 根拠は quick_start.cpp を参照 (条件数 約 4.1e5, tolerance 1e-9)
  real(c_double), parameter :: err_limit = 1.0d-6
  real(c_double) :: xi, exact, e, err_local, err_max
  logical :: ok

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)

  ! 行の分割: 各ランクが連続した行を持つ (端数は最終ランク). 番号は 0 始まりで計算する
  nstart = (n_global / nproc) * myrank
  if (myrank == nproc - 1) then
     nend = n_global - 1
  else
     nend = nstart + n_global / nproc - 1
  end if
  n = nend - nstart + 1

  ! 担当行の CRS (index_base = 1: 行ポインタ・列番号・nstart を 1 始まりで渡す)
  allocate(rowptr(n + 1), col(3 * n), val(3 * n), b(n), x(n))
  nnz = 0
  rowptr(1) = 1
  do i = nstart, nend
     if (i > 0) then
        nnz = nnz + 1; col(nnz) = i;     val(nnz) = -1.0d0     ! 列 i-1 (0 始まり) = i (1 始まり)
     end if
     nnz = nnz + 1;    col(nnz) = i + 1; val(nnz) =  2.0d0
     if (i < n_global - 1) then
        nnz = nnz + 1; col(nnz) = i + 2; val(nnz) = -1.0d0
     end if
     rowptr(i - nstart + 2) = nnz + 1
  end do
  b = 1.0d0
  x = 0.0d0

  ! 共役勾配法 (前処理なし, 倍精度)
  call parcel_options_init(opt)
  opt%index_base    = 1
  opt%tolerance     = 1.0d-9
  opt%max_iteration = 10000
  rc = parcel_solve_linear(PARCEL_SOLVER_PCG, opt, MPI_COMM_WORLD, n, nstart + 1, n_global, nnz, &
       rowptr, col, val, b, x, res)

  ! 厳密解との比較. 行列は境界の外の値を 0 とした (Dirichlet 境界) 差分 -x(i-1) + 2 x(i) - x(i+1) = 1 なので,
  !   厳密解は 0 始まりの大域番号 i で x = (i + 1) (n_global - i) / 2. 全成分の相対誤差の最大を各ランクの担当行で求め,
  !   ランクの最大を集める
  err_local = 0.0d0
  do i = nstart, nend
     xi = dble(i)
     exact = 0.5d0 * (xi + 1.0d0) * (dble(n_global) - xi)
     e = abs(x(i - nstart + 1) - exact) / exact
     if (e /= e) e = huge(e)                ! NaN は最大の値として扱う (不合格にする)
     err_local = max(err_local, e)
  end do
  call MPI_Allreduce(err_local, err_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
  ok = rc == PARCEL_OK .and. res%converged_reason > 0 .and. err_max <= err_limit
  if (myrank == 0) then
     write(*, '(a,i0,a,i0,a,i0,a,es10.3,a,es10.3)') 'rc = ', rc, ', reason = ', res%converged_reason, &
          ', iterations = ', res%iterations, ', residual = ', res%residual, ', true residual = ', res%true_residual
     write(*, '(a,f12.6,a,f12.6,a)') 'x(1) = ', x(1), ' (exact ', 0.5d0 * dble(n_global), ')'
     if (ok) then
        write(*, '(a,es9.3,a,es7.1,a)') 'check: max relative error = ', err_max, ' (limit ', err_limit, ') PASS'
     else
        write(*, '(a,es9.3,a,es7.1,a)') 'check: max relative error = ', err_max, ' (limit ', err_limit, ') FAIL'
     end if
  end if
  call MPI_Finalize(ierr)
  if (.not. ok) stop 1
end program quick_start_f
