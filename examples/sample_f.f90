! SPDX-License-Identifier: LGPL-3.0-only
! Copyright (c) 日本原子力研究開発機構 システム計算科学センター
! =========================================================================
! sample_f.f90 : PARCEL_SYCL Fortran インタフェース (parcel_sycl_mod) のサンプル兼検証
!
! sample_c.c と同じ内容を Fortran で行う。
!   3 次元 7 点ポアソン行列 (中心 -6, 隣接 1, ディリクレ境界, NX^3 行) に対し
!   7 種のソルバすべてを呼び、結果を独立に検証して PASS / FAIL を出す。
!   pcagmres は Q を陽に作らない版 (cagmres_q_mode = PARCEL_CAGMRES_Q_IMPLICIT) も
!   CholeskyQR と対称行列用のモーメント法 (PARCEL_QR_CAGMRES_CHOLESKYQR、前処理なし・s 偶数) の 2 通り試す。
!     線形 : 相対残差 ||b - Ax|| / ||b||
!     固有値 : 解析固有値 λ = -2 Σ_d (1 - cos(k_d π/(NX+1)))
!   ks_arnoldi は shift-invert (σ = -0.15) も試す。旧 API run_parcel_sycl_f も呼ぶ。
!   最初に版 (parcel_get_version / parcel_version_string) も確かめる (計 12 項目)。
!   各ソルバの PASS には、ライブラリが返す収束状態 (parcel_result) が収束
!   (converged_reason > 0) であることも含める。
!
!   CRS は Fortran 流の 1 始まり (opt%index_base = 1) で渡す。
!   行はランク r に重み (r+1) で不均等に配る。失敗があれば終了コード 1。
!
!   mpirun -n 3 ./sample_f [NX]      (NX 既定 16)
!
!   MPI なし (ラッパを -DPARCEL_MPI=0 でビルド) では、このファイルは変えずに
!   sample_f_mpi_stub.f90 (1 プロセス用の mpi モジュールの代用) と一緒に gfortran で組む。
!
!   行数・非零数・rowptr / col は API の型に合わせて integer(parcel_int_kind) (= c_int64_t) で持つ。
!   MPI の個数 (counts / displs) は既定の integer。
! =========================================================================
module sample_f_util
  use mpi
  use parcel_sycl_mod, only: parcel_result, parcel_int_kind
  implicit none
  integer, allocatable :: counts(:), displs(:)
  real(8), allocatable :: full(:)
  integer :: myrank, nprocs, n_fail = 0

  interface
     ! 旧 API (v1.1 以前との互換)。CRS は 0 始まり。
     !   n / nnz / crsCol / crsRow_ptr は 64 ビット (C の parcel_int)
     subroutine run_parcel_sycl(mpi_comm_f, n, nnz, crsA, crsCol, crsRow_ptr, vecx, vecb) &
          bind(c, name="run_parcel_sycl_f")
       use iso_c_binding
       implicit none
       integer(c_int) mpi_comm_f
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nnz
       real(c_double) crsA(*)
       integer(c_int64_t) crsCol(*)
       integer(c_int64_t) crsRow_ptr(*)
       real(c_double) vecx(*)
       real(c_double) vecb(*)
     end subroutine run_parcel_sycl
  end interface

contains

  ! 大域行 [rstart, rstart+nrow-1] (1 始まり) の CRS を 1 始まりで作る。
  !   col が未割り付けなら非零数だけ数える。
  subroutine poisson_rows(nx, rstart, nrow, rowptr, nnz, col, val)
    integer, intent(in)  :: nx
    integer(parcel_int_kind), intent(in)  :: rstart, nrow
    integer(parcel_int_kind), intent(out) :: rowptr(nrow+1), nnz
    integer(parcel_int_kind), optional, intent(out) :: col(:)
    real(8), optional, intent(out) :: val(:)
    integer(parcel_int_kind) :: r, g, ix, iy, iz
    nnz = 0
    rowptr(1) = 1
    do r = 1, nrow
       g  = rstart + r - 1            ! 1 始まり大域行
       ix = mod(g-1, nx)
       iy = mod((g-1)/nx, nx)
       iz = (g-1)/(nx*nx)
       if(iz > 0)      call put(g - nx*nx, 1.0d0)
       if(iy > 0)      call put(g - nx,    1.0d0)
       if(ix > 0)      call put(g - 1,     1.0d0)
                       call put(g,        -6.0d0)
       if(ix < nx - 1) call put(g + 1,     1.0d0)
       if(iy < nx - 1) call put(g + nx,    1.0d0)
       if(iz < nx - 1) call put(g + nx*nx, 1.0d0)
       rowptr(r+1) = nnz + 1
    end do
  contains
    subroutine put(c, v)
      integer(parcel_int_kind), intent(in) :: c
      real(8), intent(in) :: v
      nnz = nnz + 1
      if(present(col)) then
         col(nnz) = c
         val(nnz) = v
      end if
    end subroutine put
  end subroutine poisson_rows

  ! y = A x (x を全体に集めてから自分の行を掛ける。1 始まり CRS)
  subroutine spmv(n, rowptr, col, val, x, y)
    integer(parcel_int_kind), intent(in) :: n, rowptr(:), col(:)
    real(8), intent(in) :: val(:), x(:)
    real(8), intent(out) :: y(:)
    integer(parcel_int_kind) :: i, k
    integer :: ierr
    call MPI_Allgatherv(x, int(n), MPI_DOUBLE_PRECISION, full, counts, displs, &
         MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierr)
    do i = 1, n
       y(i) = 0.0d0
       do k = rowptr(i), rowptr(i+1) - 1
          y(i) = y(i) + val(k) * full(col(k))
       end do
    end do
  end subroutine spmv

  real(8) function gdot(n, a, b)
    integer(parcel_int_kind), intent(in) :: n
    real(8), intent(in) :: a(:), b(:)
    real(8) :: s
    integer :: ierr
    s = sum(a(1:n) * b(1:n))
    call MPI_Allreduce(s, gdot, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  end function gdot

  ! 全解析固有値を |λ - σ| の降順 (desc) / 昇順で並べる
  subroutine analytic_eigs(nx, sigma, desc, ref)
    integer, intent(in) :: nx
    real(8), intent(in) :: sigma
    logical, intent(in) :: desc
    real(8), intent(out) :: ref(:)
    real(8) :: c(nx), pi
    integer :: i, j, l, p
    pi = acos(-1.0d0)
    do i = 1, nx
       c(i) = 2.0d0 * (1.0d0 - cos(i * pi / (nx + 1)))
    end do
    p = 0
    do i = 1, nx
       do j = 1, nx
          do l = 1, nx
             p = p + 1
             ref(p) = -(c(i) + c(j) + c(l))
          end do
       end do
    end do
    call sort_by_key(ref, sigma, desc)
  end subroutine analytic_eigs

  ! 挿入ソート (n = NX^3 = 4096 程度なので十分)
  subroutine sort_by_key(a, sigma, desc)
    real(8), intent(inout) :: a(:)
    real(8), intent(in) :: sigma
    logical, intent(in) :: desc
    integer :: i, j
    real(8) :: t
    do i = 2, size(a)
       t = a(i)
       j = i - 1
       do while(j >= 1)
          if(desc) then
             if(abs(a(j) - sigma) >= abs(t - sigma)) exit
          else
             if(abs(a(j) - sigma) <= abs(t - sigma)) exit
          end if
          a(j+1) = a(j)
          j = j - 1
       end do
       a(j+1) = t
    end do
  end subroutine sort_by_key

  ! res を渡したときは収束状態も判定に含め、次の行に表示する
  subroutine report(name, rc, pass_in, detail, res)
    character(*), intent(in) :: name, detail
    integer, intent(in) :: rc
    logical, intent(in) :: pass_in
    type(parcel_result), intent(in), optional :: res
    logical :: pass
    pass = pass_in .and. (rc == 0)
    if(present(res)) pass = pass .and. (res%converged_reason > 0)
    if(.not. pass) n_fail = n_fail + 1
    if(myrank == 0) then
       if(pass) then
          write(*,'(a,a22,a,i3,2x,a)') '[sample_f] ', name, ' PASS  rc=', rc, trim(detail)
       else
          write(*,'(a,a22,a,i3,2x,a)') '[sample_f] ', name, ' FAIL  rc=', rc, trim(detail)
       end if
       if(present(res)) then
          write(*,'(13x,a,i0,a,i0,a,es10.3,a,es10.3,a,i0,a,i0)') 'converged_reason = ', res%converged_reason, &
               ' (> 0 で収束)  its = ', res%iterations, '  residual = ', res%residual, &
               '  true_residual = ', res%true_residual, '  ks_nconv = ', res%ks_nconv, &
               '  linsolve_failures = ', res%linsolve_failures
       end if
       flush(6)
    end if
  end subroutine report

end module sample_f_util


program sample_f
  use mpi
  use iso_c_binding
  use parcel_sycl_mod
  use sample_f_util
  implicit none

  integer :: nx, ierr, r, s, j, k, nev, mode, rc, n_multi
  integer(parcel_int_kind) :: n_global, n_local, nstart, nnz_local, i
  integer(8) :: w, acc
  integer(parcel_int_kind), allocatable :: rowptr(:), col(:), rowptr0(:), col0(:)
  real(8), allocatable :: val(:), b(:), x(:), ax(:), v(:), ev(:), ref(:), got(:)
  real(8) :: bnorm, rel, err, res, t0, t1, maxim, maxres, maxerr, err_top, best, sigma, key
  real(8), parameter :: lin_tol = 1.0d-9, lin_check = 1.0d-7
  type(parcel_options) :: opt
  type(parcel_result)  :: pres
  character(len=256) :: detail, arg
  integer :: lin_solvers(6)
  character(len=20) :: lin_names(6)
  logical :: desc, pass
  integer(c_int) :: vmaj, vmin, vpat
  integer :: vst
  character(len=32) :: vstr, vnum, vexp

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  nx = 16
  if(command_argument_count() >= 1) then
     call get_command_argument(1, arg)
     read(arg, *) nx
  end if
  n_global = int(nx, parcel_int_kind)**3

  ! Fortran の型と C の struct の大きさを照合
  if(.not. parcel_options_check()) then
     if(myrank == 0) write(*,*) '[sample_f] parcel_options の大きさが C と一致しない'
     call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if
  if(.not. parcel_result_check()) then
     if(myrank == 0) write(*,*) '[sample_f] parcel_result の大きさが C と一致しない'
     call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if

  ! ---- 不均等分割: ランク r の重み r+1 ----
  allocate(counts(0:nprocs-1), displs(0:nprocs-1), full(n_global))
  w = int(nprocs, 8) * (nprocs + 1) / 2
  acc = 0
  do r = 0, nprocs - 1
     displs(r) = int(n_global * acc / w)
     acc = acc + r + 1
     counts(r) = int(n_global * acc / w) - displs(r)
  end do
  n_local = counts(myrank)
  nstart  = displs(myrank) + 1          ! 1 始まり

  allocate(rowptr(n_local+1))
  call poisson_rows(nx, nstart, n_local, rowptr, nnz_local)
  allocate(col(nnz_local), val(nnz_local))
  call poisson_rows(nx, nstart, n_local, rowptr, nnz_local, col, val)

  if(myrank == 0) write(*,'(a,i0,a,i0,a,i0,a,*(1x,i0))') '[sample_f] NX=', nx, ' n_global=', n_global, &
       ' nprocs=', nprocs, ' rows/rank:', counts
  allocate(b(n_local), x(n_local), ax(n_local))

  ! ================= 版 =================
  ! 番号と文字列が同じ版を指すか。PARCEL_EXPECT_VERSION を与えればその値とも比べる (回帰試験の生成器が与える)
  rc = parcel_get_version(vmaj, vmin, vpat)
  call parcel_version_string(vstr)
  write(vnum, '(i0,a,i0,a,i0)') vmaj, '.', vmin, '.', vpat
  call get_environment_variable('PARCEL_EXPECT_VERSION', vexp, status=vst)
  if(vst /= 0) vexp = ''
  pass = (trim(vstr) == trim(vnum)) .and. (len_trim(vexp) == 0 .or. trim(vstr) == trim(vexp))
  if(len_trim(vexp) == 0) vexp = '-'
  write(detail, '(7a)') 'parcel_version_string = ', trim(vstr), '  parcel_get_version = ', trim(vnum), &
       '  (expect ', trim(vexp), ')'
  call report('version', rc, pass, detail)

  ! ================= 線形ソルバ =================
  ! 最後の 2 つはどちらも pcagmres の Q を陽に作らない版 (QR の選び方だけ違う。下の設定を参照)
  lin_solvers = (/ PARCEL_SOLVER_PCG, PARCEL_SOLVER_PBICGSTAB, PARCEL_SOLVER_PGMRES, PARCEL_SOLVER_PCAGMRES, &
                   PARCEL_SOLVER_PCAGMRES, PARCEL_SOLVER_PCAGMRES /)
  lin_names   = (/ 'pcg                 ', 'pbicgstab           ', 'pgmres              ', &
                   'pcagmres            ', 'pcagmres(implicit)  ', 'pcagmres(impl,sym)  ' /)
  do i = 1, n_local
     b(i) = 1.0d0 + 0.1d0 * sin(0.37d0 * (nstart - 1 + i - 1))
  end do
  bnorm = sqrt(gdot(n_local, b, b))

  do s = 1, 6
     call parcel_options_init(opt)
     opt%index_base    = 1
     opt%tolerance     = lin_tol
     opt%max_iteration = 10000
     opt%precon_algo   = PARCEL_PRECON_BLOCKJACOBI
     ! ライブラリの既定値によらず、このサンプルで使う設定を明示する
     opt%gmres_restart   = 8
     opt%gmres_qr_algo   = PARCEL_QR_MGS
     opt%cagmres_s       = 8
     opt%cagmres_t       = 1
     opt%cagmres_qr_algo = PARCEL_QR_MGS
     if(s == 5) then
        ! Q を陽に作らない版: CholeskyQR。t = 2 でブロック (k > 0) の直交化も通す
        opt%cagmres_t       = 2
        opt%cagmres_q_mode  = PARCEL_CAGMRES_Q_IMPLICIT
        opt%cagmres_qr_algo = PARCEL_QR_CHOLESKYQR
     else if(s == 6) then
        ! モーメント法 (IMPLICIT 専用): A が対称であること (ここでは成り立つ)、前処理なし、s が偶数が必要
        opt%precon_algo     = PARCEL_PRECON_NONE
        opt%cagmres_t       = 2
        opt%cagmres_q_mode  = PARCEL_CAGMRES_Q_IMPLICIT
        opt%cagmres_qr_algo = PARCEL_QR_CAGMRES_CHOLESKYQR
     end if
     if(s == 1) opt%verbose = 1
     x = 0.0d0
     t0 = MPI_Wtime()
     rc = parcel_solve_linear(lin_solvers(s), opt, MPI_COMM_WORLD, n_local, nstart, n_global, &
          nnz_local, rowptr, col, val, b, x, pres)
     t1 = MPI_Wtime()
     call spmv(n_local, rowptr, col, val, x, ax)
     ax = b - ax
     rel = sqrt(gdot(n_local, ax, ax)) / bnorm
     write(detail,'(a,es10.3,a,es8.1,a,f8.2,a)') '||b-Ax||/||b|| = ', rel, ' (<= ', lin_check, ')', t1-t0, ' s'
     call report(trim(lin_names(s)), rc, rel <= lin_check, detail, pres)
  end do

  ! ================= 固有値ソルバ =================
  allocate(ref(n_global))

  ! ---- power: 絶対値最大 ----
  call parcel_options_init(opt)
  opt%index_base    = 1
  opt%tolerance     = 1.0d-10
  opt%max_iteration = 20000
  allocate(ev(2), v(n_local))
  t0 = MPI_Wtime()
  rc = parcel_solve_eigen(PARCEL_SOLVER_POWER, opt, MPI_COMM_WORLD, n_local, nstart, n_global, &
       nnz_local, rowptr, col, val, ev, v, pres)
  t1 = MPI_Wtime()
  call analytic_eigs(nx, 0.0d0, .true., ref)
  err = abs(ev(1) - ref(1)) / abs(ref(1))
  write(detail,'(a,f16.12,a,f16.12,a,es9.2,a,f8.2,a)') 'lambda = ', ev(1), '  exact = ', ref(1), &
       '  rel.err = ', err, ' (<= 1e-6)', t1-t0, ' s'
  call report('power', rc, err <= 1.0d-6, detail, pres)

  ! ---- inverse_power: 絶対値最小 (内部 pgmres + ブロックヤコビ) ----
  call parcel_options_init(opt)
  opt%index_base           = 1
  opt%tolerance            = 1.0d-10
  opt%max_iteration        = 200
  opt%linsolve_algo        = PARCEL_LINSOLVE_PGMRES
  opt%linsolve_tolerance   = 1.0d-11
  opt%linsolve_max_iteration = 5000
  opt%linsolve_restart     = 30
  opt%linsolve_precon_algo = PARCEL_PRECON_BLOCKJACOBI
  t0 = MPI_Wtime()
  rc = parcel_solve_eigen(PARCEL_SOLVER_INVERSE_POWER, opt, MPI_COMM_WORLD, n_local, nstart, n_global, &
       nnz_local, rowptr, col, val, ev, v, pres)
  t1 = MPI_Wtime()
  call analytic_eigs(nx, 0.0d0, .false., ref)
  err = abs(ev(1) - ref(1)) / abs(ref(1))
  call spmv(n_local, rowptr, col, val, v, ax)
  ax = ax - ev(1) * v
  res = sqrt(gdot(n_local, ax, ax)) / (abs(ev(1)) * sqrt(gdot(n_local, v, v)))
  write(detail,'(a,f16.12,a,f16.12,a,es9.2,a,es8.1,a,f8.2,a)') 'lambda = ', ev(1), '  exact = ', ref(1), &
       '  rel.err = ', err, ' (<= 1e-6)  resid = ', res, ' ', t1-t0, ' s'
  call report('inverse_power', rc, err <= 1.0d-6, detail, pres)
  deallocate(ev, v)

  ! ---- ks_arnoldi: (1) 絶対値最大側 nev 個, (2) shift-invert σ=-0.15 近傍 nev 個 ----
  !   判定は sample_c.c と同じ (各 Ritz 値の最近接解析値、端の値、残差)
  sigma = -0.15d0
  do mode = 1, 2
     call parcel_options_init(opt)
     opt%index_base     = 1
     opt%tolerance      = 1.0d-8
     opt%ks_m           = 30
     opt%ks_k_keep      = 10
     opt%ks_nev         = 4
     opt%ks_max_restart = 100
     if(mode == 2) then
        opt%shift_enable         = 1
        opt%shift_sigma          = sigma
        opt%linsolve_tolerance   = 1.0d-11
        ! 内部 pgmres の上限。既定 (0) は ks_arnoldi の既定 2000 になるので、5000 を明示する
        opt%linsolve_max_iteration = 5000
        opt%linsolve_restart     = 30
        opt%linsolve_precon_algo = PARCEL_PRECON_BLOCKJACOBI
     end if
     k   = parcel_eigen_count(PARCEL_SOLVER_KS_ARNOLDI, opt)
     nev = opt%ks_nev
     allocate(ev(2*k), v(n_local*k), got(nev))
     t0 = MPI_Wtime()
     rc = parcel_solve_eigen(PARCEL_SOLVER_KS_ARNOLDI, opt, MPI_COMM_WORLD, n_local, nstart, n_global, &
          nnz_local, rowptr, col, val, ev, v, pres)
     t1 = MPI_Wtime()
     if(mode == 1) then
        key = 0.0d0;  desc = .true.
     else
        key = sigma;  desc = .false.
     end if
     call analytic_eigs(nx, key, desc, ref)
     maxim = 0.0d0; maxres = 0.0d0; maxerr = 0.0d0
     do j = 1, nev
        got(j) = ev(2*j-1)
        if(.not. (abs(ev(2*j)) <= maxim)) maxim = abs(ev(2*j))
        call spmv(n_local, rowptr, col, val, v((j-1)*n_local+1 : j*n_local), ax)
        ax = ax - got(j) * v((j-1)*n_local+1 : j*n_local)
        res = sqrt(gdot(n_local, ax, ax)) / (abs(got(j)) * &
             sqrt(gdot(n_local, v((j-1)*n_local+1 : j*n_local), v((j-1)*n_local+1 : j*n_local))))
        if(.not. (res <= maxres)) maxres = res
        best = minval(abs(got(j) - ref))
        err = best / abs(got(j))
        if(.not. (err <= maxerr)) maxerr = err
     end do
     call sort_by_key(got, key, desc)
     err_top = abs(got(1) - ref(1)) / abs(ref(1))
     n_multi = 0
     do j = 1, nev
        if(abs(got(j) - ref(j)) <= 1.0d-6 * abs(ref(j))) n_multi = n_multi + 1
     end do
     pass = (maxerr <= 1.0d-6) .and. (err_top <= 1.0d-6) .and. (maxim <= 1.0d-8) .and. (maxres <= 1.0d-6)
     write(detail,'(i0,a,es9.2,a,es9.2,a,es8.1,a,es8.1,a,f8.2,a)') nev, ' eigs: nearest rel.err = ', maxerr, &
          ', extreme rel.err = ', err_top, ' (<= 1e-6)  max|Im| = ', maxim, '  resid = ', maxres, &
          ' (<= 1e-6) ', t1-t0, ' s'
     if(mode == 1) then
        call report('ks_arnoldi', rc, pass, detail, pres)
     else
        call report('ks_arnoldi(shift-inv)', rc, pass, detail, pres)
     end if
     if(myrank == 0) then
        do j = 1, nev
           write(*,'(13x,i2,2x,f16.12,a,f16.12)') j, got(j), '  exact(sorted) ', ref(j)
        end do
        write(*,'(13x,a,i0,a,i0,a)') 'multiplicity reproduced: ', n_multi, ' / ', nev, ' (reference only)'
     end if
     deallocate(ev, v, got)
  end do

  ! ================= 旧 API (互換): 0 始まり CRS で渡す =================
  allocate(rowptr0(n_local+1), col0(nnz_local))
  rowptr0 = rowptr - 1
  col0    = col - 1
  x = 0.0d0
  call run_parcel_sycl(MPI_COMM_WORLD, n_local, nnz_local, val, col0, rowptr0, x, b)
  call spmv(n_local, rowptr, col, val, x, ax)
  ax = b - ax
  rel = sqrt(gdot(n_local, ax, ax)) / bnorm
  write(detail,'(a,es10.3,a,es8.1,a)') '||b-Ax||/||b|| = ', rel, ' (<= ', lin_check, ')'
  call report('legacy run_parcel_sycl_f', 0, rel <= lin_check, detail)

  if(myrank == 0) then
     if(n_fail == 0) then
        write(*,'(a)') '[sample_f] ALL PASS (0 failed)'
     else
        write(*,'(a,i0,a)') '[sample_f] FAILED (', n_fail, ' failed)'
     end if
  end if
  deallocate(rowptr, col, val, b, x, ax, ref, rowptr0, col0, counts, displs, full)
  call MPI_Finalize(ierr)
  if(n_fail /= 0) stop 1
end program sample_f
