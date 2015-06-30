#include "kernel_bw.h"
#include "constants_bw.h"
#include <stdio.h>

//---------------------------------------------------------------------------//
// Kernels
//---------------------------------------------------------------------------//
__global__ void EM_betaB_alphabeta(float *beta_d, 
		float *B_d, 
		float *betaB_d,  
		float *alpha_d,
		float *alpha_beta_d,
		const int N,
		int current, 
		int previous)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < N) {
		betaB_d[idx]      = beta_d[previous + idx] * B_d[previous + idx];
		alpha_beta_d[idx] = beta_d[current + idx]  * alpha_d[current + idx];
	}
}


__global__ void EM_alphabeta_update_gamma(float *alpha_beta_d, 
		float *gamma_d,
		float *ll_d, 
		const int N, 
		unsigned int current)
{
	unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < N){
		gamma_d[current + idx] = alpha_beta_d[idx] / ll_d[0];
	}
}


/// ConstA : alpha_d
/// ConstB : betaB_d 
__global__ void EM_A_mul_alphabetaB(float *a_d, 
		float *A_alphabetaB_d,
		float *blk_result_d,
		const int N) 
{

	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	uint gx = blockIdx.x * blockDim.x + lx;
	uint gy = blockIdx.y * blockDim.y + ly;

	float data;

	uint outID = gy * N + gx;
	volatile __shared__ float lds[256];

	// localsize: 16 x 16
	// alphabetaB[i][j] = alpha[i] * betaB[j];
	// A[i][j] .* alphabetaB[i][j];

	// data = A_alphabetaB[gy * N + gx] = A[gy * N + gx] * alpha[current + gy] * betaB[gx];
	data = A_alphabetaB_d[outID] = a_d[outID] * ConstA[gy] * ConstB[gx];

	// lds[ly][lx]
	uint index = ly * TILE + lx;
	lds[index] = data;

	__syncthreads();

	//reduction
	if(lx < 8) {lds[index] += lds[index + 8];}
	if(lx < 4) {lds[index] += lds[index + 4];}
	if(lx < 2) {lds[index] += lds[index + 2];}
	if(lx < 1) {lds[index] += lds[index + 1];}
	if(lx == 0 && ly == 0){
		// output block id
		int id = blockIdx.y * gridDim.x + blockIdx.x;

		// sum up the 1st column in the lds
		blk_result_d[id] = lds[0] + lds[16] + lds[32] + lds[48] 
			+ lds[64] + lds[80] + lds[96] + lds[112]
			+ lds[128] + lds[144] + lds[160] + lds[176] 
			+ lds[192] + lds[208] + lds[224] + lds[240];
	}
}

__global__ void EM_update_xisum(float *A_alphabetaB_d,
		float *xi_sum_d,
		const float sum,
		const int N) 
{
	uint gx = blockIdx.x * blockDim.x + threadIdx.x;
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint outID = gy * N + gx;
	xi_sum_d[outID] += A_alphabetaB_d[outID] / sum;
}


__global__ void EM_alphabeta(float *beta_d, 
		float *alpha_d,
		float *alpha_beta_d,
		const int N)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		alpha_beta_d[idx] = beta_d[idx] * alpha_d[idx];
	}
}



// expected_A     = mk_stochastic(xi_sum);
// sum along each row and scale rowwise
__global__ void EM_expect_A(float *xi_sum_d,
		float *expect_A_d,
		const int N) 
{
	uint gx = blockIdx.x * blockDim.x + threadIdx.x;
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	__shared__ float lds[256];

	// number of iterations, equal to the column groups, because A is square 
	size_t m =  gridDim.y;

	int i, col;
	float data;
	size_t offset = gy * N;

	// load 1st time
	data = xi_sum_d[offset + gx];
	//printf("(%d,%d) \n", gy, gx);

	//#pragma unroll
	for(i = 1 ; i < m ; ++i){
		//col = lx + 16 * i;  
		col = gx + i * TILE;  
		data += xi_sum_d[offset + col];
	}

	lds[ly*TILE + lx]= data;

	__syncthreads();

	// sum across rows
	if( gx == 0) // only 16 threads are alive now 
	{
		int start = ly * TILE;
		data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
			+ lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
			+ lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
			+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
		if (data == 0.f) data = 1.f; 
		lds[start] = data;
	}

	__syncthreads();

	for(i = 0 ; i < m ; ++i){
		col = gx + i * TILE;  
		expect_A_d[offset + col] = xi_sum_d[offset + col]/lds[ly * TILE];
	}

}


__global__ void EM_transpose(float *A,
		float *At,
		const int height, 	// T
		const int width)
{

	__shared__ float lds[272]; // (16 +1) x 16

	// read the matrix tile into shared memory
	uint xIndex = blockIdx.x * TILE + threadIdx.x;
	uint yIndex = blockIdx.y * TILE + threadIdx.y;
	uint lidx = threadIdx.x; // col  
	uint lidy = threadIdx.y; // row 

	if((xIndex < width) && (yIndex < height))
	{
		size_t index_in = yIndex * width + xIndex;
		lds[lidy * (TILE + 1) + lidx] = A[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TILE + threadIdx.x;
	yIndex = blockIdx.x * TILE + threadIdx.y;


	if((xIndex < height) && (yIndex < width))
	{
		size_t index_out = yIndex * height + xIndex;
		At[index_out] = lds[lidx * (TILE + 1) + lidy];
	}

}


__global__ void EM_gammastatesum(float *gammaT,
		float *gamma_state_sum,
		const int N,
		const int T)
{
	// gammaT :  N x T	
	__shared__ float lds[272]; // 16 x 17 

	uint gx = blockIdx.x * blockDim.x + threadIdx.x;
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	size_t m = T / TILE; 

	int i, col;
	float data;
	size_t offset = gy * T;

	// load 1st time
	data = gammaT[offset + gx];

	//#pragma unroll
	for(i = 1 ; i < m ; ++i){
		//col = lx + 16 * i;  
		col = i * TILE + gx;  
		data += gammaT[offset + col];
	}

	lds[ly*(TILE+1) + lx]= data;

	__syncthreads();

	if( gx == 0) // only 16 threads are alive now 
	{
		int start = ly * (TILE+1);
		data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
			+ lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
			+ lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
			+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
		gamma_state_sum[gy] = data;
	}
}

__global__ void EM_gammaobs(
		float *observationsT, // D x T
		float *gamma_obs,
		const int T)
{
	uint gx = blockIdx.x * blockDim.x + threadIdx.x;// col
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint id = gy * T + gx;
	gamma_obs[id] = observationsT[id] * bufferT[gx];
}


__global__ void EM_gammaobs_streams(
		float *gammaT,
		float *observationsT, // D x T
		float *gamma_obs,
		const int T)
{
	uint gx = blockIdx.x * blockDim.x + threadIdx.x;// col
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint id = gy * T + gx;
	gamma_obs[id] = observationsT[id] * gammaT[gx]; // w/o bufferT
}


__global__ void EM_expectmu(float *gamma_obs, // D x T
		const int hs,
		float *expect_mu, // N x D
		const int T, 
		const uint current)
{
	// D x T	
	// row-wise sum 
	__shared__ float lds[272]; // 16 x 16 

	uint gx = blockIdx.x * blockDim.x + threadIdx.x;
	uint gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	int m = T / TILE;  // devide column T into m TILE-trunks

	int i, col;
	float data;

	uint offset = gy * T;

	// load 1st time
	data = gamma_obs[offset + gx];

	//#pragma unroll
	for(i = 1 ; i < m ; ++i){
		//col = lx + 16 * i;  
		col = i * TILE + gx;  
		data += gamma_obs[offset + col];
	}

	lds[ly*(TILE+1) + lx]= data;

	__syncthreads();

	if( gx == 0) // only 16 threads are alive now for each block
	{
		int start = ly * (TILE+1);
		data =  lds[start]      + lds[start + 1]  + lds[start + 2]  + lds[start + 3] 
			+ lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] 
			+ lds[start + 8]  + lds[start + 9]  + lds[start + 10] + lds[start + 11] 
			+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
		expect_mu[current + gy] = data / gamma_state_sumC[hs];
	}
}


__global__ void EM_expectsigma_dev(
		float *gamma_obs,
		float *observations,	
		const int hs,
		float *expect_sigma_sym,
		const int D,
		const int T)
{
	// C = A x B
	// C , expect_sigma_sym 
	// A , gamma_obs 
	// B , observations

	// (DxT) (TxD) will produce DxD 
	__shared__ float lds_a[72]; // 8 x 9 
	__shared__ float lds_b[72]; // 

	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int nx = T / 8;
	int Col =  bx * 8 + lx; // global col index for output
	int Row =  by * 8 + ly; // global row index for output

	float sum = 0.f;        
	int m;

	for ( m = 0; m < nx ; ++m)
	{
		lds_a[ly * 9 + lx] = gamma_obs[Row * T + m * 8 + lx];        
		lds_b[ly * 9 + lx] = observations[(m * 8 + ly) * D + Col];        

		__syncthreads();

		// matrix mul on LDS
		// a x b
		int kk;
#pragma unroll
		for ( kk = 0; kk < 8; ++kk) 
		{
			sum += lds_a[ly * 9 + kk] * lds_b[kk * 9 + lx];
		}

		__syncthreads();

	}

	// sum is the mm result of gamma_obs * obs_t
	// sum * gamma_state_sum(s) - expect_mu(s) * expect_mu(s)'
	expect_sigma_sym[Row * D + Col] = sum / gamma_state_sumC[hs] 
		- expect_mu_state[Row] * expect_mu_state[Col];
}


__global__ void EM_expectsigma_dev_streams(
		float *expect_mu,
		float *gamma_obs,
		float *observations,	
		const int hs,
		float *expect_sigma_sym,
		const int D,
		const int T)
{
	// (DxT) (TxD) will produce DxD 
	__shared__ float lds_a[72]; // 8 x 9 
	__shared__ float lds_b[72]; // 

	uint lx = threadIdx.x; // col  
	uint ly = threadIdx.y; // row 

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int nx = T / 8;
	int Col =  bx * 8 + lx; // global col index for output
	int Row =  by * 8 + ly; // global row index for output

	float sum = 0.f;        
	int m;

	for ( m = 0; m < nx ; ++m)
	{
		lds_a[ly * 9 + lx] = gamma_obs[Row * T + m * 8 + lx];        
		lds_b[ly * 9 + lx] = observations[(m * 8 + ly) * D + Col];        

		__syncthreads();

		// matrix mul on LDS
		// a x b
		int kk;
#pragma unroll
		for ( kk = 0; kk < 8; ++kk) 
		{
			sum += lds_a[ly * 9 + kk] * lds_b[kk * 9 + lx];
		}

		__syncthreads();

	}

	// sum is the mm result of gamma_obs * obs_t
	// sum * gamma_state_sum(s) - expect_mu(s) * expect_mu(s)'
	expect_sigma_sym[Row * D + Col] = sum / gamma_state_sumC[hs] 
		- expect_mu[Row] * expect_mu[Col];
}
__global__ void EM_update_expectsigma(
		float *expect_sigma,	
		float *expect_sigma_sym,
		const int width,
		uint start)
{	
	// read the matrix tile into shared memory
	uint gcol = blockIdx.x * blockDim.x + threadIdx.x;
	uint grow = blockIdx.y * blockDim.y + threadIdx.y;
	
	uint idx = grow * width + gcol;

	if(grow > gcol)
	{
		// use triu to write tril 
		expect_sigma[start + idx] = expect_sigma_sym[gcol * width + grow];	
	}
	else if (grow < gcol)
	{
		expect_sigma[start + idx] = expect_sigma_sym[idx];	
	}
	else // (grow == gcol)
	{
		// ensure positive semidefiniteness
		expect_sigma[start + idx] =  expect_sigma_sym[idx] + 0.01f;	
	}


}


//---------------------------------------------------------------------------//
// Functions 
//---------------------------------------------------------------------------//
void em_betaB_alphabeta(float *beta_d, 
		float *B_d, 
		float *betaB_d,  
		float *alpha_d,
		float *alpha_beta_d,
		const int N,
		int current, 
		int previous)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256; 

	// Calculate beta * B and alpha * beta                                  
	EM_betaB_alphabeta <<< grid, block >>> (beta_d,                         
			B_d,                            
			betaB_d,                        
			alpha_d,                        
			alpha_beta_d,                   
			N,                              
			current,                        
			previous);                      

}

// Update gamma                                                         
void em_alphabeta_update_gamma(float *alpha_beta_d, 
		float *gamma_d,
		float *ll_d, 
		const int N, 
		unsigned int current)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256; 

	EM_alphabeta_update_gamma <<< grid, block >>> (alpha_beta_d, 
			gamma_d, 
			ll_d, 
			N, 
			current);
}

void em_A_mul_alphabetaB(float *alpha_d,
		float *betaB_d,
		size_t bytes_n,
		float *a_d, 
		float *A_alphabetaB_d,
		float *blk_result_d,
		const int N)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE);

	// Copy data from global to constant mem                                
	cudaMemcpyToSymbol(ConstA, alpha_d, bytes_n, 0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(ConstB, betaB_d, bytes_n, 0, cudaMemcpyDeviceToDevice);

	// A .*  (alpha * betaB') 	
	// vector mul vector, then element-wise matrix mul
	EM_A_mul_alphabetaB <<< grid, block >>> (a_d, 
			A_alphabetaB_d, 
			blk_result_d, 
			N);
}


/// update xisum
void em_update_xisum(float *A_alphabetaB_d,
		float *xi_sum_d,
		float sum,
		const int N)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE);

	EM_update_xisum <<< grid, block >>> (A_alphabetaB_d, xi_sum_d, sum, N);
}


/// alpha * beta
void em_alphabeta(float *beta_d, 
		float *alpha_d,
		float *alpha_beta_d,
		const int N)
{
	int block = 256;                                                                
	int grid  = (N + 255)/256;  

	EM_alphabeta <<< grid, block >>> (beta_d, 
			alpha_d,
			alpha_beta_d, 
			N);

}

void em_expect_A(float *xi_sum_d,
		float *expect_A_d,
		const int N) 
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid(1, (N + TILE - 1)/TILE);

	EM_expect_A <<< grid, block >>> (xi_sum_d, 
			expect_A_d, 
			N); 
}


void em_transpose(float *A,
		float *At,
		const int T, 	// height
		const int N)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid((N + TILE - 1)/TILE, (T + TILE - 1)/TILE);

	// transpose gamma: from (T x N) to (N x T)                                 
	EM_transpose <<< grid, block >>> (A, At, T, N); 	
}

void em_gammastatesum(float *gammaT_d,
		float *gamma_state_sum_d,
		const int N,
		const int T)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid((T + TILE - 1)/TILE, (N + TILE - 1)/TILE);

	EM_gammastatesum <<< grid, block >>> (gammaT_d, gamma_state_sum_d, N, T);

	// copy gamma_state_sum to constant memory (read-only)                      
	cudaMemcpyToSymbol(gamma_state_sumC, 
			gamma_state_sum_d, 
			sizeof(float) * N, 
			0, 
			cudaMemcpyDeviceToDevice);

}



void em_gammaobs(
		float *gammaT_d,
		size_t bytes_t,
		float *observationsT_d,
		float *gamma_obs_d,
		const int T,
		const int D)
{
	/// copy to constant buffer
	cudaMemcpyToSymbol(bufferT, gammaT_d, bytes_t, 0, cudaMemcpyDeviceToDevice);

	dim3 block(TILE, TILE);                                                           
	dim3 grid((T + TILE - 1)/TILE, (D + TILE - 1)/TILE);

	EM_gammaobs <<< grid, block >>> (observationsT_d, gamma_obs_d, T);
}


void em_gammaobs_streams(
		float *gammaT_d,
		size_t bytes_t,
		float *observationsT_d,
		float *gamma_obs_d,
		const int T,
		const int D,
		int sid,
		cudaStream_t *streams)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid((T + TILE - 1)/TILE, (D + TILE - 1)/TILE);

	EM_gammaobs_streams <<< grid, block, 0, streams[sid] >>> (
			gammaT_d,
			observationsT_d, 
			gamma_obs_d, 
			T);
}



void em_expectmu(
		float *gamma_obs_d, // D x T
		const int hs,
		float *expect_mu_d, // N x D
		const int T, 
		const int D,
		const uint current)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid(1, (D + TILE - 1)/TILE);

	/// use the gamma_state_sumC in em_gammastatesum()
	EM_expectmu <<< grid, block >>> (
			gamma_obs_d, 
			hs, 
			expect_mu_d, 
			T, 
			current);


	cudaMemcpyToSymbol(
			expect_mu_state,	// constant 
			&expect_mu_d[current], 
			sizeof(float) * D, 
			0, 
			cudaMemcpyDeviceToDevice);
}


/// cuda stream version
void em_expectmu_streams(
		float *gamma_obs_d, // D x T
		const int hs,
		float *expect_mu_d, // N x D
		const int T, 
		const int D,
		const uint current,
		int sid,
		cudaStream_t  *streams)
{
	dim3 block(TILE, TILE);                                                           
	dim3 grid(1, (D + TILE - 1)/TILE);

	/// use the same kernel 
	EM_expectmu <<< grid, block, 0, streams[sid] >>> (
			gamma_obs_d, 
			hs, 
			expect_mu_d, 
			T, 
			current);
}



void em_expectsigma_dev(
		float *gamma_obs_d,
		float *observations_d,	
		const int hs,
		float *expect_sigma_sym_d,
		const int D,
		const int T)
{
	/// gamma_obs * obs' / gamma_state_sum(s) - exp_mu(:, s) * exp_mu(:, s)'
	/// gamma_obs	: D x T
	/// obs			: T x D 
	dim3 block_12(8, 8);                                                            
	dim3 grid_12((D+7)/8, (D+7)/8);

	EM_expectsigma_dev <<< grid_12, block_12 >>> (
			gamma_obs_d, 
			observations_d, 
			hs, 
			expect_sigma_sym_d, 
			D, 
			T);  	
}

void em_expectsigma_dev_streams(
		float *expect_mu_d,
		float *gamma_obs_d,
		float *observations_d,	
		const int hs,
		float *expect_sigma_sym_d,
		const int D,
		const int T,
		int sid,
		cudaStream_t *streams)
{
	/// gamma_obs * obs' / gamma_state_sum(s) - exp_mu(:, s) * exp_mu(:, s)'
	/// gamma_obs	: D x T
	/// obs			: T x D 
	dim3 block_12(8, 8);                                                            
	dim3 grid_12((D+7)/8, (D+7)/8);

	EM_expectsigma_dev_streams <<< grid_12, block_12, 0, streams[sid] >>> (
			expect_mu_d,
			gamma_obs_d, 
			observations_d, 
			hs, 
			expect_sigma_sym_d, 
			D, 
			T);  	
}



void em_update_expectsigma(
		float *expect_sigma_d,	
		float *expect_sigma_sym_d,
		const int D,
		uint start)
{
	/// symmtrize function                                                       
	/// 1) use upper trianglular part to update the lower triangular part
	/// 2) add 0.01 on the diagnal element
	dim3 block_13(TILE, TILE);                                                      
	dim3 grid_13((D+TILE-1)/TILE, (D+TILE-1)/TILE);
	
	EM_update_expectsigma <<< grid_13, block_13 >>> (expect_sigma_d, 
			expect_sigma_sym_d,
			D, 
			start);          

}


void em_update_expectsigma_streams(
		float *expect_sigma_d,	
		float *expect_sigma_sym_d,
		const int D,
		uint start,
		int sid,
		cudaStream_t *streams)
{
	/// symmtrize function                                                       
	/// 1) use upper trianglular part to update the lower triangular part
	/// 2) add 0.01 on the diagnal element
	dim3 block_13(TILE, TILE);                                                      
	dim3 grid_13((D+TILE-1)/TILE, (D+TILE-1)/TILE);
	
	/// use default kernel
	EM_update_expectsigma <<< grid_13, block_13, 0, streams[sid] >>> (expect_sigma_d, 
			expect_sigma_sym_d,
			D, 
			start);          

}
