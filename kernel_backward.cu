#include "kernel_backward.h"

//---------------------------------------------------------------------------//
// Kernels
//---------------------------------------------------------------------------//
__global__ void BK_update_beta(float *beta_d, float *B_d, float *betaB_d, const int N)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < N) {
		betaB_d[idx] = B_d[idx] * beta_d[idx];
	}
}


__global__ void BK_scaling( const int N, float *beta_d, float *ll_d)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < N) {
		beta_d[idx] /= ll_d[0];
	}
}



//---------------------------------------------------------------------------//
// Functions 
//---------------------------------------------------------------------------//
void bk_update_beta(float *beta_d, 
                    float *B_d, 
                    float *betaB_d, 
                    const int N)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;

	BK_update_beta <<< grid, block >>> (beta_d, 
	                                    B_d, 
										betaB_d, 
										N);
}


void bk_scaling(const int N, 
                float *beta_d, 
                float *ll_d)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;	

	BK_scaling <<< grid, block >>> (N, 
	                                beta_d,
									ll_d);  
}
