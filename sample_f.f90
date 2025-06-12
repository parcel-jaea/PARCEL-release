module parcel_sycl_wrapper

  interface
     subroutine run_parcel_sycl( &
          mpi_comm_f, &
          n,nnz, &
          crsA, &
          crsCol, &
          crsRow_ptr, &
          vecx, &
          vecb &
          ) &
          bind(c,name="run_parcel_sycl_f")
       implicit none
       integer mpi_comm_f
       integer(kind=4),value :: n
       integer(kind=4),value :: nnz
       real*8 crsA(*)
       integer(kind=4) crsCol(*)
       integer(kind=4) crsRow_ptr(*)
       real*8 vecx(*)
       real*8 vecb(*)

     end subroutine run_parcel_sycl
  end interface
contains
  
end module parcel_sycl_wrapper

program main
  use mpi
  use parcel_sycl_wrapper
  implicit none
  integer(kind=4) nx,ny,nz
  integer(kind=4) nx0,ny0,nz0
  
  integer(kind=4),parameter :: gnx = 16
  integer(kind=4),parameter :: gny = 16
  integer(kind=4),parameter :: gnz = 16
  
  integer(kind=4) gn,n
  integer(kind=4) nnz
  integer(kind=4) istart
  
  real*8,allocatable :: crsA(:)
  integer(kind=4),allocatable :: crsCol(:)
  integer(kind=4),allocatable :: crsRow_ptr(:)

  real*8,allocatable :: vecx(:)
  real*8,allocatable :: vecb(:)

  integer ierr
  integer npes,myrank

  integer i
  integer mpi_comm_f

  mpi_comm_f = MPI_COMM_WORLD
  
  ! MPI Init
  call MPI_Init(ierr)
  call MPI_Comm_size(mpi_comm_f,npes,ierr)
  call MPI_Comm_rank(mpi_comm_f,myrank,ierr)

  nx     = gnx
  ny     = gny
  nz     = gnz/npes
  
  nx0    = 1
  ny0    = 1 
  nz0    = 1 + nz*myrank

  gn     = gnx*gny*gnz
  n      = nx * ny * nz
  istart = 1+n*myrank

  call make_poisson3d_crs_get_nnz( &
       nx,ny,nz,   &
       nx0,ny0,nz0,&
       gnx,gny,gnz,&
       nnz, &
       istart)
  
  allocate(crsA(nnz))
  allocate(crsCol(nnz))
  allocate(crsRow_ptr(n+1))
  allocate(vecx(n))
  allocate(vecb(n))

  vecx = 0.0d0
  vecb = 1.0d0
  call make_poisson3d_crs( &
       nx,ny,nz,   &
       nx0,ny0,nz0,&
       gnx,gny,gnz,&
       crsA,crsCol,crsRow_ptr,n,nnz, &
       istart)
  
  call run_parcel_sycl( &
       mpi_comm_f, &
       n,nnz, &
       crsA, &
       crsCol, &
       crsRow_ptr, &
       vecx, &
       vecb &
       ) 

  deallocate(crsA)
  deallocate(crsCol)
  deallocate(crsRow_ptr)
  deallocate(vecx)
  deallocate(vecb)
  
  call MPI_Finalize(ierr); 
  
end program main
