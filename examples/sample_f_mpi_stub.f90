! =========================================================================
! sample_f_mpi_stub.f90 : MPI なしのビルド (PARCEL_MPI=0) で sample_f.f90 をそのまま組むための
!                         1 プロセス用の mpi モジュールの代用
!
! sample_f.f90 と quick_start_f.f90 が使う分 (MPI_Init / Comm_rank / Comm_size / Abort / Wtime / Allgatherv /
! Allreduce / Finalize と定数 4 個) だけを 1 プロセスとして実装する。ライブラリとは関係しない
! (ラッパは -DPARCEL_MPI=0 でコミュニケータの引数を使わない)。
!
!   gfortran -c sample_f_mpi_stub.f90      -> mpi.mod ができ、sample_f.f90 の "use mpi" がこれを読む
!   gfortran -c ../capi/parcel_sycl_mod.f90 sample_f.f90   (examples/build_examples.sh の PARCEL_MPI=0 を参照)
! MPI のコンパイララッパ (mpif90) では使わない (本物の mpi モジュールと名前が衝突する)。
!
! SPDX-License-Identifier: LGPL-3.0-only
! Copyright (c) 日本原子力研究開発機構 システム計算科学センター
! =========================================================================
module mpi
  implicit none
  integer, parameter :: MPI_COMM_WORLD       = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 0
  integer, parameter :: MPI_SUM              = 0
  integer, parameter :: MPI_MAX              = 1

contains

  subroutine MPI_Init(ierr)
    integer, intent(out) :: ierr
    ierr = 0
  end subroutine MPI_Init

  subroutine MPI_Finalize(ierr)
    integer, intent(out) :: ierr
    ierr = 0
  end subroutine MPI_Finalize

  subroutine MPI_Comm_rank(comm, rank, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: rank, ierr
    rank = 0
    ierr = 0
  end subroutine MPI_Comm_rank

  subroutine MPI_Comm_size(comm, size, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: size, ierr
    size = 1
    ierr = 0
  end subroutine MPI_Comm_size

  subroutine MPI_Abort(comm, errorcode, ierr)
    integer, intent(in)  :: comm, errorcode
    integer, intent(out) :: ierr
    ierr = 0
    write(0, '(a,i0)') 'MPI_Abort (1 process stub): errorcode = ', errorcode
    error stop
  end subroutine MPI_Abort

  double precision function MPI_Wtime()
    integer(8) :: c, r
    call system_clock(c, r)
    MPI_Wtime = dble(c) / dble(r)
  end function MPI_Wtime

  ! 1 プロセス: 自分の分をそのまま受信バッファの displs(1) の位置に写す
  subroutine MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, ierr)
    real(8), intent(in)    :: sendbuf(*)
    real(8), intent(inout) :: recvbuf(*)
    integer, intent(in)    :: sendcount, sendtype, recvcounts(*), displs(*), recvtype, comm
    integer, intent(out)   :: ierr
    recvbuf(displs(1)+1 : displs(1)+sendcount) = sendbuf(1:sendcount)
    ierr = 0
  end subroutine MPI_Allgatherv

  ! 1 プロセス: スカラー 1 個をそのまま写す (sample_f.f90 と quick_start_f.f90 の使い方だけ。演算 op によらない)
  subroutine MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm, ierr)
    real(8), intent(in)  :: sendbuf
    real(8), intent(out) :: recvbuf
    integer, intent(in)  :: count, datatype, op, comm
    integer, intent(out) :: ierr
    recvbuf = sendbuf
    ierr = 0
  end subroutine MPI_Allreduce

end module mpi
