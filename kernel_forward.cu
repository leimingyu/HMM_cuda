#include "kernel_forward.h"

//---------------------------------------------------------------------------//
// Kernels
//---------------------------------------------------------------------------//
__global__ void FWD_init_alpha(float *b_d, 
                               float *pi_d, 
							   int N, 
							   float *alpha_d, 
							   float *ones_d, 
		                       float *beta_d)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < N) {
		alpha_d[idx] = pi_d[idx] * b_d[idx];
		beta_d[idx] = ones_d[idx] = 1.0f; // for backward
	}
}


__global__ void FWD_scaling(int N, float *alpha_d, float *scale_factor, int t)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (idx < N) {
		alpha_d[idx] /= scale_factor[t];
	}
}


__global__ void FWD_calc_alpha( int N , float *alpha_d , float *b_d )
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (idx < N) {
		alpha_d[idx] *= b_d[idx];
	}
}


__global__ void FWD_sum_ll(const int T, float *ll_d)
{
	unsigned int lid = threadIdx.x;
	unsigned int gid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	// T = 64
	__shared__ float sm[64];

	if (gid < T){
		sm[lid] = log10(ll_d[gid]);
	}

	__syncthreads();

	//reduction
	if (lid < 32) {
		volatile float* smem = sm;
		smem[lid] += smem[lid + 32];
		smem[lid] += smem[lid + 16];
		smem[lid] += smem[lid +  8];
		smem[lid] += smem[lid +  4];
		smem[lid] += smem[lid +  2];
		smem[lid] += smem[lid +  1];
	}

	if (lid == 0) {
		ll_d[T] = sm[0];
	}
}


//---------------------------------------------------------------------------//
// Functions 
//---------------------------------------------------------------------------//
void fwd_init_alpha(float *b_d, 
                    float *pi_d,
					const int N,
					float *alpha_d,
					float *ones_d,
					float *beta_d)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;

	/// alpha = b * pi
	/// initialize ones_d for cublas
	/// initialize beta_d
	FWD_init_alpha <<< grid, block >>> (b_d, 
	                                    pi_d, 
										N, 
										alpha_d, 
										ones_d, 
										beta_d);
}


void fwd_scaling(const int N, 
                 float *alpha_d, 
	             float *ll_d, 
	             int t)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;

	// element-wise division                                                    
	FWD_scaling <<< grid, block >>> (N, 
	                                 alpha_d, 
									 ll_d, 
									 t); 
}

void fwd_calc_alpha(const int N, 
                    float *alpha_d, 
	                float *b_d)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;

	FWD_calc_alpha <<< grid, block >>> (N , 
	                                    alpha_d,
										b_d);
}

void fwd_sum_ll(const int T, 
		        float *ll_d) 
{
    FWD_sum_ll <<< 1, T >>> (T, ll_d);    
}
