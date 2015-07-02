#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>                                                                           
#include <helper_cuda.h>      // check errors                                                            
#include <helper_functions.h> // timer  

#include <cublas.h>		      // cublas
#include <cublas_v2.h>	      // new cublas api: pointer mode

#include "kernel_forward.h"
#include "kernel_backward.h"
#include "kernel_bw.h"

#ifndef TILE
	#define TILE 16	// 2D Kernel Tiling
#endif

#ifndef SIZE 
	#define SIZE 4096 
#endif

//---------------------------------------------------------------------------//
// CPU Parameters 
//---------------------------------------------------------------------------//
int N;		// number of hidden states
int T = 64;	// number of (overlapping) windows 
int D = 64;	// number of features
uint nstreams = 2;  // default

float *a;			// state transition probability matrix
float *b;			// emission probability matrix 
float *pi;			// prior probability
float *alpha;		// forward probability matrix
float *lll;			// log likelihood
float *blk_result;	// intermediate blk results 
float *observations;

int bytes_nn; 
int bytes_nt; 
int bytes_dt;  
int bytes_dd; 
int bytes_dn; 
int bytes_ddn; 
int bytes_t;
int bytes_d;  
int bytes_n;  
int dd;

int tileblks;
size_t bytes_tileblks;


//---------------------------------------------------------------------------//
// GPU Parameters 
//---------------------------------------------------------------------------//
int block = 256;
int grid  = (N + 255)/256;

dim3 block_3(16, 16);
dim3 grid_3(N/16, N/16);

dim3 block_6(16, 16);
dim3 grid_6(1, N/16); 

dim3 block_7(16, 16);
dim3 grid_7(N/16, T/16);

dim3 block_8(16, 16);
dim3 grid_8(T/16, N/16);

dim3 block_9(16, 16);
dim3 grid_9(D/16, T/16);

// gammaobs
dim3 block_10(16, 16);
dim3 grid_10(T/16, D/16);

// expectmu
dim3 block_11(16, 16);
dim3 grid_11(1, D/16);

// expectsigma_dev 
dim3 block_12(8, 8);
dim3 grid_12(D/8, D/8);

// cublas	
cublasStatus ret;
cublasHandle_t handle, handle1;	// handler
float alp = 1.f;
float bet = 0.f;

// forward
float *a_d;
float *b_d;
float *pi_d;
float *alpha_d;
float *ones_d;
float *ll_d;

// bk
float *beta_d;
float *betaB_d;

// em 
float *xi_sum_d;
float *alpha_beta_d;
float *gamma_d;
float *A_alphabetaB_d;
float *blk_result_d;
float *gammaT_d;
float *gamma_state_sum_d;
float *gamma_obs_d;
float *expect_mu_d;
float *expect_sigma_sym_d;
float *expect_sigma_d;

float *expect_prior_d;
float *expect_A_d;
float *observations_d;
float *observationsT_d;

// create a timer                                                                               
StopWatchInterface *timer = NULL;                                                               

// cuda streams
cudaStream_t *streams;

//---------------------------------------------------------------------------//
// Functions 
//---------------------------------------------------------------------------//
void HMM_Param();
void GPU_HMM_Forward();
void GPU_HMM_Backward();
void GPU_HMM_BaumWelch();
void Release();


//---------------------------------------------------------------------------//
// Main Program
//---------------------------------------------------------------------------//
int main(int argc, char *argv[])
{
	if(argc != 3){
		puts("Please specify the number of hidden states N and cuda streams K.");
		puts("e.g., $./gpuhmmsr N K (N should be multiples of 16)\nExit Program!");
		exit(1);
	}

	N = atoi(argv[1]);
	printf("Hidden States: %d\n", N);

	nstreams = atoi(argv[2]);
#if HQ
	printf("Use Hyper-Q feature, with %d cuda streams.\n\n", nstreams);
#else
	printf("Hyper-Q feature is not applied!\n\n");
#endif


	printf("=> Start program.\n\n");

	//-----------------------------------------------------------------------//
	// HMM Parameters
	//	a,b,pi,alpha
	//-----------------------------------------------------------------------//
	printf("(1) Initialize parameters.\n");
	HMM_Param();

	//-----------------------------------------------------------------------//
	// Forward Algorithm on GPU 
	//-----------------------------------------------------------------------//
	printf("\n");
	printf("(2) Forward Algorithm on GPU.\n");
	GPU_HMM_Forward();

	//-----------------------------------------------------------------------//
	// Backward Algorithm on GPU 
	//-----------------------------------------------------------------------//
	printf("\n");
	printf("(3) Backward Algorithm on GPU.\n");
	GPU_HMM_Backward();

	//-----------------------------------------------------------------------//
	// Baum-Welch Algorithm on GPU 
	//-----------------------------------------------------------------------//
	printf("\n");
	printf("(4) Baum-Welch Algorithm on GPU.\n");
	GPU_HMM_BaumWelch();

	//-----------------------------------------------------------------------//
	// Release resources
	//-----------------------------------------------------------------------//
	Release();
	printf("\n<= End program.\n");

	return 0;
}

void HMM_Param()
{

	//-----------------------------------------------------------------------//
	// size of arrays
	//-----------------------------------------------------------------------//
	bytes_nn  = sizeof(float) * N * N;
	bytes_nt  = sizeof(float) * N * T;
	bytes_n   = sizeof(float) * N;
	bytes_dt  = sizeof(float) * D * T;
	bytes_dd  = sizeof(float) * D * D;
	bytes_dn  = sizeof(float) * D * N ;
	bytes_ddn = sizeof(float) * D * D * N ;
	bytes_t   = sizeof(float) * T;
	bytes_d   = sizeof(float) * D;
	bytes_n   = sizeof(float) * N;
	dd        = D * D;

	tileblks = ((N+TILE-1)/TILE) * ((N+TILE-1)/TILE);
	bytes_tileblks = sizeof(float) * tileblks;


	int i, j;

	//-----------------------------------------------------------------------//
	// pinned memory
	//	a,b,pi,lll, blk_result
	//-----------------------------------------------------------------------//

	// create timer
	sdkCreateTimer(&timer);                                                                         

	// state transition probability matrix
	checkCudaErrors(cudaMallocHost((void **)&a, bytes_nn));
	for (i = 0; i < (N * N); i++) {
		a[i] = 1.0f/(float)N;
	}

	// emission probability matrix 
	checkCudaErrors(cudaMallocHost((void **)&b, bytes_nt));
	for (i = 0; i < (N * T); i++) {
		b[i] = 1.0f/(float)T;
	}

	// prior probability
	checkCudaErrors(cudaMallocHost((void **)&pi, bytes_n));
	for (i = 0; i < N; i++) {
		pi[i] = 1.0f/(float)N;
	}

	// intermediate blk results from the device
	checkCudaErrors(cudaMallocHost((void **)&blk_result, bytes_tileblks));

	// log likelihood 
	checkCudaErrors(cudaMallocHost((void **)&lll, sizeof(float)));

	// forward probability matrix
	// hint: for checking purpose
	alpha = (float *)malloc(bytes_nt);  // T x N

	// for em
	checkCudaErrors(cudaMallocHost((void **)&observations, bytes_dt)); // T x D

	for(i = 0 ; i< T ; ++i) {
		for(j = 0 ; j< D ; ++j) {
			observations[i * D + j] = (float)i + 1.f;
		}
	}

	//-----------------------------------------------------------------------//
	// GPU Memory 
	//-----------------------------------------------------------------------//

	// forward 
	checkCudaErrors(cudaMalloc((void**)&a_d, bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&b_d, bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&pi_d, bytes_n));
	checkCudaErrors(cudaMalloc((void**)&alpha_d, bytes_nt)); 
	checkCudaErrors(cudaMalloc((void**)&ones_d, bytes_n));	// for cublasdot
	checkCudaErrors(cudaMalloc((void**)&ll_d, sizeof(float)*(T + 1))); 

	// backward
	checkCudaErrors(cudaMalloc((void**)&beta_d,  bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&betaB_d, bytes_n)); 

	// EM
	checkCudaErrors(cudaMalloc((void**)&xi_sum_d,          bytes_nn)); 
	checkCudaErrors(cudaMalloc((void**)&alpha_beta_d,      bytes_n)); 
	checkCudaErrors(cudaMalloc((void**)&gamma_d,           bytes_nt)); 
	checkCudaErrors(cudaMalloc((void**)&A_alphabetaB_d,    bytes_nn)); 
	checkCudaErrors(cudaMalloc((void**)&blk_result_d,      bytes_tileblks)); 
	checkCudaErrors(cudaMalloc((void**)&gammaT_d,          bytes_nt)); 
	checkCudaErrors(cudaMalloc((void**)&gamma_state_sum_d, bytes_n)); 
	checkCudaErrors(cudaMalloc((void**)&gamma_obs_d,       bytes_dt)); 

	checkCudaErrors(cudaMalloc((void**)&expect_prior_d,    bytes_n)); 
	checkCudaErrors(cudaMalloc((void**)&expect_A_d,        bytes_nn)); 
	checkCudaErrors(cudaMalloc((void**)&observations_d,    bytes_dt)); 
	checkCudaErrors(cudaMalloc((void**)&observationsT_d,   bytes_dt)); 

	checkCudaErrors(cudaMalloc((void**)&expect_mu_d,       	bytes_dn)); 
	checkCudaErrors(cudaMalloc((void**)&expect_sigma_sym_d, bytes_dd)); 
	checkCudaErrors(cudaMalloc((void**)&expect_sigma_d,     bytes_ddn)); 

	/// cuda streams
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++)                                          
		checkCudaErrors(cudaStreamCreate(&(streams[i])));                       
}


void Release()
{
	// cpu
	free(alpha);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(pi);
	cudaFreeHost(lll);
	cudaFreeHost(blk_result);
	cudaFreeHost(observations);

	// gpu : fo
	checkCudaErrors(cudaFree(a_d));
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(pi_d));
	checkCudaErrors(cudaFree(alpha_d));
	checkCudaErrors(cudaFree(ones_d));
	checkCudaErrors(cudaFree(ll_d));

	// gpu : bk 
	checkCudaErrors(cudaFree(beta_d));
	checkCudaErrors(cudaFree(betaB_d));

	// gpu : em
	checkCudaErrors(cudaFree(xi_sum_d));
	checkCudaErrors(cudaFree(alpha_beta_d));
	checkCudaErrors(cudaFree(gamma_d));
	checkCudaErrors(cudaFree(A_alphabetaB_d));
	checkCudaErrors(cudaFree(blk_result_d));
	checkCudaErrors(cudaFree(gammaT_d));
	checkCudaErrors(cudaFree(gamma_state_sum_d));
	checkCudaErrors(cudaFree(gamma_obs_d));

	checkCudaErrors(cudaFree(expect_prior_d));
	checkCudaErrors(cudaFree(expect_A_d));
	checkCudaErrors(cudaFree(observations_d));
	checkCudaErrors(cudaFree(observationsT_d));

	checkCudaErrors(cudaFree(expect_mu_d));
	checkCudaErrors(cudaFree(expect_sigma_sym_d));
	checkCudaErrors(cudaFree(expect_sigma_d));

	for (int i = 0; i < nstreams; i++)                                          
		checkCudaErrors(cudaStreamDestroy(streams[i]));                         
}


void GPU_HMM_Forward()
{
	// Initialize cublas
	ret = cublasInit();
	if (ret != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf (stderr, "ERROR: CUBLAS Initialization failure\n");
		exit(EXIT_FAILURE);
	}

	ret  = cublasCreate(&handle);
	ret  = cublasCreate(&handle1);

	// Make sure the data remain on the device 
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	// Move data to device: non-blocking
	checkCudaErrors(cudaMemcpyAsync(a_d,  a,  bytes_nn, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(b_d,  b,  bytes_nt, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(pi_d, pi, bytes_n,  cudaMemcpyHostToDevice));

	//printf("Calculate Log-likelihood\n");
	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStartTimer(&timer);  

	// alpha = b * pri
	// beta_d initialization for backward algo
	fwd_init_alpha (b_d, 
	                pi_d, 
					N, 
					&alpha_d[0], 
					ones_d, 
					&beta_d[(T-1)*N]);

	// sum(alpha)
	ret = cublasSdot(handle, N, &alpha_d[0], 1, ones_d, 1, &ll_d[0]);
	if (ret != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
		exit(EXIT_FAILURE);
	}

	// element-wise division
	fwd_scaling (N , &alpha_d[0], ll_d, 0);

	int frm;
	int current, previous;

	for (frm = 1; frm < T; ++frm) 
	{
		current  = frm * N; 
		previous = current - N;

		// a' * alpha
		// auto transposed due to the column major thing
		ret = cublasSgemv(handle1, CUBLAS_OP_N, 
				N, N,
				&alp, 
				a_d, N, 
				&alpha_d[previous], 1,
				&bet, 
				&alpha_d[current], 1);

		if (ret != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		// b * (a' * alpha) 
		fwd_calc_alpha (N, &alpha_d[current], &b_d[current]);

		// the likelihood for current window
		ret = cublasSdot(handle, N, 
				&alpha_d[current], 1, 
				ones_d, 1, 
				&ll_d[frm]);

		if (ret != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		fwd_scaling (N , &alpha_d[current], ll_d, frm);
	}
	// parallel reduction on the likelihood
	fwd_sum_ll (T, ll_d);

	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStopTimer(&timer);                                                                           
	double runtime = sdkGetTimerValue(&timer);                                                      

	printf("    Elapsed Time = %lf ms\n", runtime);      

	// copy the log likelihood back to host
	//checkCudaErrors(cudaMemcpyAsync(lll, &ll_d[T], sizeof(float), cudaMemcpyDeviceToHost));
	// check 
	//printf("log likehood = %f\n", lll[0]);
}

void GPU_HMM_Backward()
{
	// beta_d is pre-computed in forward step

	int j;
	int current, previous;

	/// start timer
	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStartTimer(&timer);  

	// Calcuate backwards 
	for(j = T-2; j >= 0; --j)
	{
		current  = j * N;
		previous = current + N;

		// betaB = beta(t) * b
		bk_update_beta (&beta_d[previous], &b_d[previous], betaB_d, N);

		// beta(t-1) = a * betaB
		ret = cublasSgemv(handle1, CUBLAS_OP_T, 
				N, N, 
				&alp,
				a_d, N, 
				betaB_d, 1, 
				&bet, 
				&beta_d[current], 1);

		if (ret != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		// sum up
		ret = cublasSdot(handle, N, 
				&beta_d[current], 1, 
				ones_d, 1, 
				&ll_d[0]); // use ll_d[0] to save the sum

		if (ret != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		// normalise
		bk_scaling (N , &beta_d[current], ll_d);
	}

	// stop timer
	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStopTimer(&timer);                                                                           

	printf("    Elapsed Time = %lf ms\n", sdkGetTimerValue(&timer));      
}

void GPU_HMM_BaumWelch()
{
	/// start timer
	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStartTimer(&timer);  

	// clear the data for xi_sum
	checkCudaErrors(cudaMemset(xi_sum_d, 0, bytes_nn));

	float sum;
	int window, i;
	int current, previous;

	// symmtrize function
	//int blk_rows = D/16;
	//uint blknum = blk_rows * (blk_rows + 1) / 2;

	// update_expectsigma
	//dim3 block_13(16, 16);
	//dim3 grid_13(1, blknum);// instead of N, lauch enough blocks

	for(window=0; window < (T-1) ; ++window)
	{
		current = window * N;
		previous = current + N;

		// Calculate beta * B and alpha * beta
		em_betaB_alphabeta(beta_d, 
		                   b_d, 
						   betaB_d, 
						   alpha_d, 
						   alpha_beta_d, 
						   N, 
				           current, 
						   previous);

		// sum up alpha_beta using cublas
		ret = cublasSdot(handle, N,
				alpha_beta_d, 1, 
				ones_d, 1,
				&ll_d[0]);

		if (ret != CUBLAS_STATUS_SUCCESS) 
		{
			fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
			exit(EXIT_FAILURE);
		}

		// Update gamma
		em_alphabeta_update_gamma (alpha_beta_d, gamma_d, ll_d, N, current);

		// A .*  (alpha * betaB')
		em_A_mul_alphabetaB (&alpha_d[current],
		                     betaB_d,
							 bytes_n,
		                     a_d, 
		                     A_alphabetaB_d, 
							 blk_result_d, 
							 N);

		checkCudaErrors(cudaMemcpyAsync(blk_result, blk_result_d, bytes_tileblks,
					    cudaMemcpyDeviceToHost));

		sum = 0.f;	
#pragma unroll
		for(i=0; i<tileblks; ++i)
		{
			sum += blk_result[i];
		}	

		// Normalise A_alphabetaB and add up to xi_sum 
		//EM_update_xisum <<< grid_3, block_3 >>> (A_alphabetaB_d, xi_sum_d, sum, N);
		em_update_xisum (A_alphabetaB_d, 
		                 xi_sum_d, 
						 sum, 
						 N);
	}

	current = previous;

	//EM_alphabeta <<< grid, block >>> (&beta_d[current], &alpha_d[current], alpha_beta_d, N);
	em_alphabeta (&beta_d[current], 
	              &alpha_d[current], 
				  alpha_beta_d, 
				  N);
	ret = cublasSdot(handle, N, alpha_beta_d, 1, ones_d, 1, &ll_d[0]);
	if (ret != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
		exit(EXIT_FAILURE);
	}

	em_alphabeta_update_gamma (alpha_beta_d, gamma_d, ll_d, N, current);

	// expected_prior = gamma(:, 1);
	checkCudaErrors(cudaMemcpy(expect_prior_d, &gamma_d[0], bytes_n, cudaMemcpyDeviceToDevice));

	// expected_A     = mk_stochastic(xi_sum);
	//EM_expect_A <<< grid_6, block_6 >>> (xi_sum_d, expect_A_d, N);
	em_expect_A (xi_sum_d, expect_A_d, N);

	// transpose gamma: from (T x N) to (N x T) 
	// EM_transpose <<< grid_7, block_7 >>> (gamma_d, gammaT_d, T, N);
	em_transpose (gamma_d, gammaT_d, T, N);

	// gamma_state_sum = sum(gamma, 2); 
	// T x N for gamma_d
	// sum row on gammaT_d(N x T)
	em_gammastatesum (gammaT_d, gamma_state_sum_d, N, T);

	// copy gamma_state_sum to constant memory (read-only)
	// cudaMemcpyToSymbol(gamma_state_sumC, gamma_state_sum_d, bytes_n, 0, cudaMemcpyDeviceToDevice);

	// hint: while gpu is running, these "observations" operations can be concurrently run on CPU
	checkCudaErrors(cudaMemcpyAsync(observations_d, 
	                                observations, 
				                    bytes_dt, 
									cudaMemcpyHostToDevice));

	// EM_transpose<<< grid_9, block_9 >>> (observations_d, observationsT_d, T, D);
	em_transpose (observations_d, observationsT_d, T, D);

	int hs;

#if HQ
	//-----------------------------------------------------------------------//
	// Hyper-Q
	//-----------------------------------------------------------------------//
	
	int streamid;
	for(hs = 0 ; hs < N; ++hs)
	{
		streamid = hs % nstreams;	

		// fixeme : use no constant memory due to increased demand using K nstreams
		em_gammaobs_streams(
				&gammaT_d[hs * T], 
				bytes_t, 
				observationsT_d, 
				gamma_obs_d, 
				T,
				D,
				streamid,
				streams);

		current = hs * D;

		em_expectmu_streams (
				gamma_obs_d, 
				hs, 
				expect_mu_d, 
				T, 
				D,
				current,
				streamid,
				streams);

		em_expectsigma_dev_streams(
				&expect_mu_d[current], 	//expect_mu_state
				gamma_obs_d, 
				observations_d, 
				hs, 
				expect_sigma_sym_d, 
				D, 
				T,
				streamid,
				streams);

		/// symmetrize(expect_sigma)
		em_update_expectsigma_streams(
				expect_sigma_d, 
				expect_sigma_sym_d, 
				D, 
				hs * dd,
				streamid,
				streams);


	}

#else
	for(hs = 0 ; hs < N; ++hs)
	{
		em_gammaobs(
				&gammaT_d[hs * T], 
				bytes_t, 
				observationsT_d, 
				gamma_obs_d, 
				T,
				D);

		current = hs * D;

		em_expectmu (
				gamma_obs_d, 
				hs, 
				expect_mu_d, 
				T, 
				D,
				current);

		/// epxect_sigma
		/// gamma_obs * obs' / gamma_state_sum(s) - exp_mu(:, s) * exp_mu(:, s)'
		em_expectsigma_dev(
				gamma_obs_d, 
				observations_d, 
				hs, 
				expect_sigma_sym_d, 
				D, 
				T);

		/// symmetrize(expect_sigma)
		em_update_expectsigma(
				expect_sigma_d, 
				expect_sigma_sym_d, 
				D, 
				hs * dd);
	}
#endif

	checkCudaErrors(cudaDeviceSynchronize());                                                       
	sdkStopTimer(&timer);                                                                           
	double runtime = sdkGetTimerValue(&timer);                                                      

	printf("    Elapsed Time = %lf ms\n", runtime);      
}
