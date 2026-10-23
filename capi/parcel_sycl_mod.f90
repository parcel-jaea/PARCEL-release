! =========================================================================
! parcel_sycl_mod.f90
!
! SPDX-License-Identifier: LGPL-3.0-only
! Copyright (c) 日本原子力研究開発機構 システム計算科学センター
!
! PARCEL_SYCL の Fortran インタフェース (C API parcel_sycl_c.h と同じ関数)。
!
!   use parcel_sycl_mod
!   integer(parcel_int_kind) :: n_local, nstart, n_global, nnz_local     ! 64 ビット
!   integer(parcel_int_kind), allocatable :: rowptr(:), col(:)
!   type(parcel_options) :: opt
!   call parcel_options_init(opt)          ! 既定値
!   opt%tolerance   = 1.0d-9
!   opt%index_base  = 1                    ! nstart / col を 1 始まりで渡すとき
!   type(parcel_result)  :: res
!   ierr = parcel_solve_linear(PARCEL_SOLVER_PGMRES, opt, MPI_COMM_WORLD, &
!                              n_local, nstart, n_global, nnz_local, &
!                              rowptr, col, val, b, x, res)
!   if(res%converged_reason <= 0) ...      ! 未収束 (PARCEL_DIVERGED_*)
!   ierr = parcel_solve_eigen (PARCEL_SOLVER_KS_ARNOLDI, opt, MPI_COMM_WORLD, &
!                              n_local, nstart, n_global, nnz_local, &
!                              rowptr, col, val, eigval, eigvec)   ! res は省略可 (C には NULL が渡る)
!
! コミュニケータは Fortran の整数ハンドル (mpi モジュールの MPI_COMM_WORLD など) を渡す。
! ラッパを MPI なし (-DPARCEL_MPI=0) でビルドしたときは値を使わない (0 など任意の整数)。
!   このモジュールは MPI の有無によらず同じ (1 プロセスなので nstart = index_base、n_local = n_global)。
! 各引数の意味・戻り値は parcel_sycl_c.h を参照。
!   eigval(2*k) : 実部・虚部が交互。k = parcel_eigen_count(solver, opt)
!   eigvec(n_local*k) : 列優先 (eigvec(1+(j-1)*n_local : j*n_local) が j 番目)
!
! n_local / nstart / n_global / nnz_local と rowptr(:) / col(:) は
!   integer(c_int64_t) (= integer(parcel_int_kind)、C の parcel_int)。
!   ライブラリ内部の PARCEL_INT (既定 64 ビット、-DPARCEL_INT32 で 32 ビット) によらず 64 ビット固定。
!   値渡しの引数に既定の integer (4 バイト) の変数を渡すとコンパイルエラーになるので、
!   int(n_local, parcel_int_kind) などで変換するか、変数を integer(parcel_int_kind) で宣言する。
!   配列 (rowptr, col) は integer(parcel_int_kind) で確保すること (4 バイトの配列は渡せない)。
!
! parcel_options / parcel_result は C の構造体と同じ並び (bind(C))。
! 片方を変えたらもう片方も直すこと。parcel_options_check() / parcel_result_check() で
! 大きさを照合できる。
! =========================================================================
module parcel_sycl_mod
  use iso_c_binding
  implicit none
  private :: parcel_solve_linear_c, parcel_solve_eigen_c, parcel_version_string_c

  ! 行番号・列番号・行数・非零数の種別 (C の parcel_int = int64_t)
  integer, parameter :: parcel_int_kind = c_int64_t

  ! ---- ソルバ ----
  integer(c_int), parameter :: PARCEL_SOLVER_PCG           = 0
  integer(c_int), parameter :: PARCEL_SOLVER_PBICGSTAB     = 1
  integer(c_int), parameter :: PARCEL_SOLVER_PGMRES        = 2
  integer(c_int), parameter :: PARCEL_SOLVER_PCAGMRES      = 3
  integer(c_int), parameter :: PARCEL_SOLVER_POWER         = 4
  integer(c_int), parameter :: PARCEL_SOLVER_INVERSE_POWER = 5
  integer(c_int), parameter :: PARCEL_SOLVER_KS_ARNOLDI    = 6
  ! Q を陽に作らない CA-GMRES は PARCEL_SOLVER_PCAGMRES +
  !   cagmres_q_mode = PARCEL_CAGMRES_Q_IMPLICIT + cagmres_qr_algo = PARCEL_QR_CHOLESKYQR を使う
  ! ---- 前処理 ----
  integer(c_int), parameter :: PARCEL_PRECON_NONE          = 0
  integer(c_int), parameter :: PARCEL_PRECON_POINTJACOBI   = 1
  integer(c_int), parameter :: PARCEL_PRECON_BLOCKJACOBI   = 2
  integer(c_int), parameter :: PARCEL_PRECON_JACOBI        = 4
  ! ---- QR ----
  integer(c_int), parameter :: PARCEL_QR_CGS               = 0
  integer(c_int), parameter :: PARCEL_QR_MGS               = 1
  integer(c_int), parameter :: PARCEL_QR_CHOLESKYQR        = 2
  integer(c_int), parameter :: PARCEL_QR_CAGMRES_CHOLESKYQR = 202  ! pcagmres の IMPLICIT 専用
  ! ---- CA-GMRES の直交基底 Q の作り方 (cagmres_q_mode) ----
  integer(c_int), parameter :: PARCEL_CAGMRES_Q_AUTO        = 0
  integer(c_int), parameter :: PARCEL_CAGMRES_Q_EXPLICIT    = 1
  integer(c_int), parameter :: PARCEL_CAGMRES_Q_IMPLICIT    = 2
  ! ---- 内部線形ソルバ ----
  integer(c_int), parameter :: PARCEL_LINSOLVE_PGMRES      = 0
  integer(c_int), parameter :: PARCEL_LINSOLVE_PCG         = 1
  ! ---- 戻り値 ----
  integer(c_int), parameter :: PARCEL_OK                   = 0
  integer(c_int), parameter :: PARCEL_ERR_ARG              = -1
  integer(c_int), parameter :: PARCEL_ERR_DISTRIBUTION     = -2
  integer(c_int), parameter :: PARCEL_ERR_NOT_BUILT        = -3
  integer(c_int), parameter :: PARCEL_ERR_RUNTIME          = -4
  ! ---- 収束状態 (parcel_result%converged_reason) ----
  integer(c_int), parameter :: PARCEL_CONVERGED_RTOL       = 1
  integer(c_int), parameter :: PARCEL_ITERATING            = 0
  integer(c_int), parameter :: PARCEL_DIVERGED_ITS         = -1
  integer(c_int), parameter :: PARCEL_DIVERGED_NANORINF    = -2
  integer(c_int), parameter :: PARCEL_DIVERGED_BREAKDOWN   = -3

  type, bind(C) :: parcel_options
     real(c_double) :: tolerance
     real(c_double) :: linsolve_tolerance
     real(c_double) :: shift_sigma
     integer(c_int) :: max_iteration
     integer(c_int) :: precon_algo
     integer(c_int) :: jacobi_iter
     integer(c_int) :: blockjacobi
     integer(c_int) :: scaling_matrix
     integer(c_int) :: scaling_vector
     integer(c_int) :: index_base
     integer(c_int) :: verbose
     integer(c_int) :: gmres_restart
     integer(c_int) :: gmres_qr_algo
     integer(c_int) :: cagmres_s
     integer(c_int) :: cagmres_t
     integer(c_int) :: cagmres_qr_algo
     integer(c_int) :: precon_spectrum
     integer(c_int) :: linsolve_algo
     integer(c_int) :: linsolve_max_iteration
     integer(c_int) :: linsolve_restart
     integer(c_int) :: linsolve_precon_algo
     integer(c_int) :: linsolve_jacobi_iter
     integer(c_int) :: linsolve_blockjacobi
     integer(c_int) :: ks_m
     integer(c_int) :: ks_k_keep
     integer(c_int) :: ks_nev
     integer(c_int) :: ks_max_restart
     integer(c_int) :: shift_enable
     integer(c_int) :: eigvec_seed
     ! pcagmres の Q の作り方 (C と同じく末尾)
     integer(c_int) :: cagmres_q_mode
     integer(c_int) :: ks_qr_algo      ! ks_arnoldi の直交化 PARCEL_QR_CGS (CGS2) / PARCEL_QR_MGS (MGS2)
  end type parcel_options

  ! 解いた結果の収束状態 (各項目の意味は parcel_sycl_c.h を参照)
  type, bind(C) :: parcel_result
     integer(c_int) :: converged_reason
     integer(c_int) :: iterations
     real(c_double) :: residual
     real(c_double) :: true_residual
     integer(c_int) :: ks_nconv
     integer(c_int) :: linsolve_failures
  end type parcel_result

  interface
     subroutine parcel_options_init(opt) bind(C, name="parcel_options_init")
       import :: parcel_options
       type(parcel_options), intent(out) :: opt
     end subroutine parcel_options_init

     function parcel_options_size() bind(C, name="parcel_options_size") result(sz)
       import :: c_int
       integer(c_int) :: sz
     end function parcel_options_size

     function parcel_result_size() bind(C, name="parcel_result_size") result(sz)
       import :: c_int
       integer(c_int) :: sz
     end function parcel_result_size

     ! 版の番号 (C の parcel_get_version と同じ。戻り値は PARCEL_OK)
     function parcel_get_version(major, minor, patch) bind(C, name="parcel_get_version") result(ierr)
       import :: c_int
       integer(c_int), intent(out) :: major, minor, patch
       integer(c_int) :: ierr
     end function parcel_get_version

     ! 内部用 (利用者は下の parcel_version_string を呼ぶ)。C の文字列へのポインタ
     function parcel_version_string_c() bind(C, name="parcel_version_string") result(p)
       import :: c_ptr
       type(c_ptr) :: p
     end function parcel_version_string_c

     function parcel_eigen_count(solver, opt) bind(C, name="parcel_eigen_count") result(k)
       import :: c_int, parcel_options
       integer(c_int), value      :: solver
       type(parcel_options), intent(in) :: opt
       integer(c_int) :: k
     end function parcel_eigen_count

     ! 内部用 (利用者は下の parcel_solve_linear / parcel_solve_eigen を呼ぶ)。res は C のポインタ (NULL 可)
     function parcel_solve_linear_c(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
          rowptr, col, val, b, x, res) bind(C, name="parcel_solve_linear_f") result(ierr)
       import :: c_int, c_int64_t, c_double, c_ptr, parcel_options
       integer(c_int), value            :: solver
       type(parcel_options), intent(in) :: opt
       integer(c_int), value            :: comm        ! Fortran の MPI コミュニケータ
       integer(c_int64_t), value        :: n_local, nstart, n_global, nnz_local   ! 64 ビット
       integer(c_int64_t), intent(in)   :: rowptr(*), col(*)
       real(c_double), intent(in)       :: val(*), b(*)
       real(c_double), intent(inout)    :: x(*)
       type(c_ptr), value               :: res
       integer(c_int) :: ierr
     end function parcel_solve_linear_c

     function parcel_solve_eigen_c(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
          rowptr, col, val, eigval, eigvec, res) bind(C, name="parcel_solve_eigen_f") result(ierr)
       import :: c_int, c_int64_t, c_double, c_ptr, parcel_options
       integer(c_int), value            :: solver
       type(parcel_options), intent(in) :: opt
       integer(c_int), value            :: comm
       integer(c_int64_t), value        :: n_local, nstart, n_global, nnz_local   ! 64 ビット
       integer(c_int64_t), intent(in)   :: rowptr(*), col(*)
       real(c_double), intent(in)       :: val(*)
       real(c_double), intent(inout)    :: eigval(*), eigvec(*)
       type(c_ptr), value               :: res
       integer(c_int) :: ierr
     end function parcel_solve_eigen_c
  end interface

contains

  ! res は省略可 (省略すると C 側には NULL が渡る)。
  !   BIND(C) の手続きに OPTIONAL な引数を付けるのは Fortran 2018 の機能で, 富士通 Fortran (frt) が受け付けない。
  !   C の関数は res を C のポインタで受ける内部用のインタフェース (parcel_solve_*_c) にし,
  !   利用者が呼ぶ parcel_solve_linear / parcel_solve_eigen は Fortran の手続きで包む
  !   (呼び方はこれまでと同じ。res があれば c_loc(res), 無ければ c_null_ptr を渡す)
  function parcel_solve_linear(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
       rowptr, col, val, b, x, res) result(ierr)
    integer(c_int), intent(in)       :: solver
    type(parcel_options), intent(in) :: opt
    integer(c_int), intent(in)       :: comm        ! Fortran の MPI コミュニケータ
    integer(c_int64_t), intent(in)   :: n_local, nstart, n_global, nnz_local   ! 64 ビット
    integer(c_int64_t), intent(in)   :: rowptr(*), col(*)
    real(c_double), intent(in)       :: val(*), b(*)
    real(c_double), intent(inout)    :: x(*)
    type(parcel_result), intent(out), optional, target :: res
    integer(c_int) :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (present(res)) p = c_loc(res)
    ierr = parcel_solve_linear_c(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
         rowptr, col, val, b, x, p)
  end function parcel_solve_linear

  function parcel_solve_eigen(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
       rowptr, col, val, eigval, eigvec, res) result(ierr)
    integer(c_int), intent(in)       :: solver
    type(parcel_options), intent(in) :: opt
    integer(c_int), intent(in)       :: comm
    integer(c_int64_t), intent(in)   :: n_local, nstart, n_global, nnz_local   ! 64 ビット
    integer(c_int64_t), intent(in)   :: rowptr(*), col(*)
    real(c_double), intent(in)       :: val(*)
    real(c_double), intent(inout)    :: eigval(*), eigvec(*)
    type(parcel_result), intent(out), optional, target :: res
    integer(c_int) :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (present(res)) p = c_loc(res)
    ierr = parcel_solve_eigen_c(solver, opt, comm, n_local, nstart, n_global, nnz_local, &
         rowptr, col, val, eigval, eigvec, p)
  end function parcel_solve_eigen

  ! 版の文字列 ("1.2.0" など。C の parcel_version_string と同じ値)。str が短ければ切り詰め、残りは空白
  subroutine parcel_version_string(str)
    character(len=*), intent(out) :: str
    character(kind=c_char), pointer :: cs(:)
    integer :: i
    str = ' '
    call c_f_pointer(parcel_version_string_c(), cs, [64])
    do i = 1, min(len(str), 64)
       if (cs(i) == c_null_char) exit
       str(i:i) = cs(i)
    end do
  end subroutine parcel_version_string

  ! Fortran 側の parcel_options と C 側の struct の大きさが一致するか (.true. なら一致)
  logical function parcel_options_check()
    type(parcel_options) :: t
    parcel_options_check = (int(c_sizeof(t), c_int) == parcel_options_size())
  end function parcel_options_check

  ! 同じく parcel_result (.true. なら一致)
  logical function parcel_result_check()
    type(parcel_result) :: t
    parcel_result_check = (int(c_sizeof(t), c_int) == parcel_result_size())
  end function parcel_result_check

end module parcel_sycl_mod
