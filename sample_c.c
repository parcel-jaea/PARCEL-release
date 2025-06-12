#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

extern void make_poisson3d_crs_get_nnz_( 
				       int *nx,
				       int *ny,
				       int *nz,
				       int *nx0,
				       int *ny0,
				       int *nz0,
				       int *gnx,
				       int *gny,
				       int *gnz,
				       int *nnz,
				       int *istart);
extern void make_poisson3d_crs_( 
			       int *nx,
			       int *ny,
			       int *nz,
			       int *nx0,
			       int *ny0,
			       int *nz0,
			       int *gnx,
			       int *gny,
			       int *gnz,
			       double *crsA,
			       int *crsCol,
			       int *crsRow_ptr,
			       int *n,
			       int *nnz0,
			       int *istart);
extern void run_parcel_sycl_c( 
		       MPI_Comm mpi_comm_c, 
		       int n, int nnz,
		       double *crsA, 
		       int *crsCol, 
		       int *crsRow_ptr, 
		       double *vecx, 
		       double *vecb 
			);

int main(int argc, char *argv[]) {

  int gnx = 16;
  int gny = 16;
  int gnz = 16;

  MPI_Init(&argc, &argv);
  MPI_Comm mpi_comm_c = MPI_COMM_WORLD;
  int npes,myrank;
  MPI_Comm_size(MPI_COMM_WORLD,&npes);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  
  int nx     = gnx;
  int ny     = gny;
  int nz     = gnz/npes;
  
  int nx0    = 1;
  int ny0    = 1 ;
  int nz0    = 1 + nz*myrank;

  int gn     = gnx*gny*gnz;
  int n      = nx * ny * nz;
  int istart = 1+n*myrank;
  int nnz;

  make_poisson3d_crs_get_nnz_( 
       &nx,&ny,&nz,   
       &nx0,&ny0,&nz0,
       &gnx,&gny,&gnz,
       &nnz, 
       &istart);

  double* crsA = (double*)malloc(sizeof(double)*nnz);
  int* crsCol = (int*)malloc(sizeof(int)*nnz);
  int* crsRow_ptr = (int*)malloc(sizeof(int)*(n+1));
  double* vecx = (double*)malloc(sizeof(double)*n);
  double* vecb = (double*)malloc(sizeof(double)*n);

  make_poisson3d_crs_(
		     &nx,&ny,&nz,
		     &nx0,&ny0,&nz0,
		     &gnx,&gny,&gnz,
		     crsA,crsCol,crsRow_ptr,
		     &n,&nnz, 
		     &istart);
  for(int i = 0;i<n;i++){
    vecx[i] = 0.0;
    vecb[i] = 1.0;
  }
  
  run_parcel_sycl_c( 
		    mpi_comm_c, 
		    n,nnz, 
		    crsA, 
		    crsCol, 
		    crsRow_ptr, 
		    vecx, 
		    vecb 
		     );

  free(crsA);
  free(crsCol);
  free(crsRow_ptr);
  free(vecx);
  free(vecb);

  MPI_Finalize(); 

  return 0;
}
