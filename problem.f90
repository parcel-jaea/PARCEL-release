subroutine make_poisson3d_crs_get_nnz( &
     nx,ny,nz,   &
     nx0,ny0,nz0,&
     gnx,gny,gnz,&
     nnz, &
     istart)

  implicit none
  integer(kind=4) nx,ny,nz,nx0,ny0,nz0,gnx,gny,gnz
  integer(kind=4) istart
  !  real*8 A( (nx*ny) * (gnx*gny) ) ! nx*ny :: 行

  integer(kind=4) i,j,k
  real*8 dxi,dyi,dzi
  real*8 dxi2,dyi2,dzi2
  real*8 dx,dy,dz
  integer(kind=4) nc,nw,ne,ns,nn,nb,nt
  integer(kind=4) Row,Col
  real*8  cc,cw,ce,cs,cn,cb,ct
  integer(kind=4) n
  integer(kind=4) ix,iy,iz

  integer(kind=4) nnz0
  integer(kind=4) nnz

  n=nx*ny*nz
  dx=1.0d0
  dy=1.0d0
  dz=1.0d0

  dxi=1.0d0/dx
  dyi=1.0d0/dy
  dzi=1.0d0/dz

  dxi2=dxi*dxi
  dyi2=dyi*dyi
  dzi2=dzi*dzi

  nnz=0
  !  do i=0,n-1
  i=0
  do iz=nz0,nz0+nz-1
     do iy=ny0,ny0+ny-1
        do ix=nx0,nx0+nx-1
           Row=ix + (iy-1)*gnx + (iz-1)*gnx*gny

           nc = Row
           nw = nc - 1  
           ne = nc + 1  
           ns = nc - gnx 
           nn = nc + gnx 
           nb = nc - gnx*gny
           nt = nc + gnx*gny

           if(ix.eq.1)then
              nw=-1
           endif
           if(ix.eq.gnx)then
              ne=-1
           endif

           if(iy.eq.1)then
              ns=-1
           endif
           if(iy.eq.gny)then
              nn=-1
           endif

           if(iz.eq.1)then
              nb=-1
           endif
           if(iz.eq.gnz)then
              nt=-1
           endif

           ! -----
           if(nb > 0)then
              nnz=nnz+1
           endif

           if(ns > 0)then
              nnz=nnz+1
           endif
           if(nw > 0)then
              nnz=nnz+1
           endif
           nnz=nnz+1

           if(ne > 0)then
              nnz=nnz+1
           endif
           if(nn > 0)then
              nnz=nnz+1
           endif
           if(nt > 0)then
              nnz=nnz+1
           endif

        enddo
     enddo
  enddo

end subroutine make_poisson3d_crs_get_nnz

subroutine make_poisson3d_crs( &
     nx,ny,nz,   &
     nx0,ny0,nz0,&
     gnx,gny,gnz,&
     crsA,crsCol,crsRow_ptr,n,nnz0, &
     istart)

  implicit none
  integer(kind=4) nx,ny,nz,nx0,ny0,nz0,gnx,gny,gnz
  integer(kind=4) istart
  !  real*8 A( (nx*ny) * (gnx*gny) ) ! nx*ny :: 行

  integer(kind=4) i,j,k
  real*8 dxi,dyi,dzi
  real*8 dxi2,dyi2,dzi2
  real*8 dx,dy,dz
  integer(kind=4) nc,nw,ne,ns,nn,nb,nt
  integer(kind=4) Row,Col
  real*8  cc,cw,ce,cs,cn,cb,ct
  integer(kind=4) n
  integer(kind=4) ix,iy,iz

  integer(kind=4) nnz0
  real*8  crsA(nnz0)
  integer(kind=4) crsCol(nnz0)
  integer(kind=4) crsRow_ptr(n+1)

  integer(kind=4) nnz

  n=nx*ny*nz
  dx=1.0d0
  dy=1.0d0
  dz=1.0d0

  dxi=1.0d0/dx
  dyi=1.0d0/dy
  dzi=1.0d0/dz

  dxi2=dxi*dxi
  dyi2=dyi*dyi
  dzi2=dzi*dzi

  nnz=1
  !  do i=0,n-1
  i=0
  do iz=nz0,nz0+nz-1
     do iy=ny0,ny0+ny-1
        do ix=nx0,nx0+nx-1
           ! ix=nx0 + mod(i,nx)
           ! iy=ny0 + i/nx 
           ! Row=i+1 + istart -1 

           crsRow_ptr(i+1)=nnz
           Row=ix + (iy-1)*gnx + (iz-1)*gnx*gny

           nc = Row
           nw = nc - 1  
           ne = nc + 1  
           ns = nc - gnx 
           nn = nc + gnx 
           nb = nc - gnx*gny
           nt = nc + gnx*gny

           cc=-2.0*(dxi2+dyi2+dzi2)
           cw=dxi2
           ce=dxi2
           cs=dyi2
           cn=dyi2
           cb=dzi2
           ct=dzi2
           ! --------
           ! cb = 1.14d0
           ! cs = 1.21d0
           ! cw = 1.0d0
           ! ce = 1.0d0
           ! cn = 1.0d0
           ! ct = 1.0d0
           ! cc = -6.0d0

           if(ix.eq.1)then
              nw=-1
           endif
           if(ix.eq.gnx)then
              ne=-1
           endif

           if(iy.eq.1)then
              ns=-1
           endif
           if(iy.eq.gny)then
              nn=-1
           endif

           if(iz.eq.1)then
              nb=-1
           endif
           if(iz.eq.gnz)then
              nt=-1
           endif

           ! -----
           if(nb > 0)then
              !           A(nb)
              crsA(nnz)   = cb
              crsCol(nnz) = nb
              nnz=nnz+1
           endif

           if(ns > 0)then
              !           A(ns)
              crsA(nnz)=cs
              crsCol(nnz) = ns
              nnz=nnz+1
           endif
           if(nw > 0)then
              !           A(nw)
              crsA(nnz)=cw
              crsCol(nnz) =nw
              nnz=nnz+1
           endif
           !        A(nc)
           crsA(nnz)=cc
           crsCol(nnz) = nc
           nnz=nnz+1

           if(ne > 0)then
              !           A(ne)
              crsA(nnz)=ce
              crsCol(nnz) = ne
              nnz=nnz+1
           endif
           if(nn > 0)then
              !           A(nn)
              crsA(nnz)=cn
              crsCol(nnz) = nn
              nnz=nnz+1
           endif
           if(nt > 0)then
              !           A(nt)
              crsA(nnz)=ct
              crsCol(nnz) = nt
              nnz=nnz+1
           endif

           !      write(*,*)i+1-1,ns,cs
           !      write(*,*)i+1-1,nw,cw
           !      write(*,*)i+1-1,nc,cc
           !      write(*,*)i+1-1,ne,ce
           !      write(*,*)i+1-1,nn,cn
           i=i+1
        enddo
     enddo
  enddo
  crsRow_ptr(n+1)=nnz
  do i = 1,nnz
     crsCol(i) =crsCol(i) -1
  enddo
  do i = 1,n+1
     crsRow_ptr(i) = crsRow_ptr(i) -1
  enddo

end subroutine make_poisson3d_crs

