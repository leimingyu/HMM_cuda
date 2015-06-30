#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>		// check errors
#include <cublas.h>		      	// cublas
#include <cublas_v2.h>	      	// new cublas api: pointer mode

#include "kernel_forward.h"
#include "kernel_backward.h"
#include "kernel_bw.h"

/// cpu parameters
int N = 64, T = 64, D = 64;
float alp = 1.f;                                                                
float bet = 0.f;
size_t bytes_nt  = sizeof(float) * N * T;                                          
size_t bytes_n   = sizeof(float) * N;
size_t bytes_ll  = sizeof(float) * (T + 1);
size_t bytes_nn  = sizeof(float) * N * N;
size_t bytes_dt  = sizeof(float) * D * T;                                          
size_t bytes_t   = sizeof(float) * T;
size_t bytes_dn  = sizeof(float) * D * N;
size_t bytes_dd  = sizeof(float) * D * D;
size_t bytes_ddn = sizeof(float) * D * D * N;

int    tileblks       = ((N+TILE-1)/TILE) * ((N+TILE-1)/TILE);
size_t bytes_tileblks = sizeof(float) * tileblks;

float *b;    
float *pi;   
float *alpha;
float *ones;
float *beta;
float *ll;
float *a;
float *betaB;
float *alpha_beta;
float *gma;
float *blk_result;
float *A_alphabetaB;
float *xi_sum;
float *expect_A;
float *gmaT;
float *gamma_state_sum;
float *obsT;
float *gamma_obs;
float *expect_mu;
float *obs;
float *expect_sigma_sym;
float *expect_sigma;

/// gpu memory
float *b_d; 
float *pi_d; 
float *alpha_d; 
float *ones_d; 
float *beta_d;
float *ll_d;
float *a_d;
float *betaB_d;
float *alpha_beta_d;
float *gma_d;
float *blk_result_d;
float *A_alphabetaB_d;
float *xi_sum_d;
float *expect_A_d;
float *gmaT_d;
float *gamma_state_sum_d;
float *obsT_d;
float *gamma_obs_d;
float *expect_mu_d;
float *obs_d;
float *expect_sigma_sym_d;
float *expect_sigma_d;

/// cublas
cublasStatus ret;                                                               
cublasHandle_t handle, handle1;

/// allocate resource
void allocate_data()
{
	/// cpu
	b                = (float *) malloc (bytes_nt);
	pi               = (float *) malloc (bytes_n);
	alpha            = (float *) malloc (bytes_nt);
	ones             = (float *) malloc (bytes_n);
	beta             = (float *) malloc (bytes_nt);
	ll               = (float *) malloc (bytes_ll);
	a                = (float *) malloc (bytes_nn);
	betaB            = (float *) malloc (bytes_n);
	alpha_beta       = (float *) malloc (bytes_n);
	gma              = (float *) malloc (bytes_nt);
	blk_result       = (float *) malloc (bytes_tileblks);
	A_alphabetaB     = (float *) malloc (bytes_nn);
	xi_sum           = (float *) malloc (bytes_nn);
	expect_A         = (float *) malloc (bytes_nn);
	gmaT             = (float *) malloc (bytes_nt);
	gamma_state_sum  = (float *) malloc (bytes_n);
	obsT             = (float *) malloc (bytes_dt);
	gamma_obs        = (float *) malloc (bytes_dt);
	expect_mu        = (float *) malloc (bytes_dn);
	obs              = (float *) malloc (bytes_dt);
	expect_sigma_sym = (float *) malloc (bytes_dd);
	expect_sigma     = (float *) malloc (bytes_ddn);

	/// gpu
	checkCudaErrors(cudaMalloc((void**)&b_d,               bytes_nt));                        
	checkCudaErrors(cudaMalloc((void**)&pi_d,              bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&alpha_d,           bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ones_d,            bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&beta_d,            bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ll_d,              bytes_ll));
	checkCudaErrors(cudaMalloc((void**)&a_d,               bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&betaB_d,           bytes_n));  
    checkCudaErrors(cudaMalloc((void**)&alpha_beta_d,      bytes_n));
	checkCudaErrors(cudaMalloc((void**)&gma_d,             bytes_nt)); 
	checkCudaErrors(cudaMalloc((void**)&blk_result_d,      bytes_tileblks)); 
	checkCudaErrors(cudaMalloc((void**)&A_alphabetaB_d,    bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&xi_sum_d,          bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&expect_A_d,        bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&gmaT_d,            bytes_nt)); 
	checkCudaErrors(cudaMalloc((void**)&gamma_state_sum_d, bytes_n)); 
	checkCudaErrors(cudaMalloc((void**)&obsT_d,            bytes_dt)); 
	checkCudaErrors(cudaMalloc((void**)&gamma_obs_d,       bytes_dt)); 
	checkCudaErrors(cudaMalloc((void**)&expect_mu_d,       bytes_dn)); 
	checkCudaErrors(cudaMalloc((void**)&obs_d,             bytes_dt)); 
	checkCudaErrors(cudaMalloc((void**)&expect_sigma_sym_d,bytes_dd)); 
	checkCudaErrors(cudaMalloc((void**)&expect_sigma_d,    bytes_ddn)); 

	/// cublas
	ret = cublasInit();                                                         
	if (ret != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "ERROR: CUBLAS Initialization failure\n");             
		exit(EXIT_FAILURE);                                                     
	}                                                                           
                                                                                
    ret  = cublasCreate(&handle);
    ret  = cublasCreate(&handle1);

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

/// release
void release_data()
{
	/// cpu
	free(b);
	free(pi);
	free(alpha);
	free(ones);
	free(beta);
	free(ll);
	free(a);
	free(betaB);
	free(alpha_beta);
	free(gma);
	free(blk_result);
	free(A_alphabetaB);
	free(xi_sum);
	free(expect_A);
	free(gmaT);
	free(gamma_state_sum);
	free(obsT);
	free(gamma_obs);
	free(expect_mu);
	free(obs);
	free(expect_sigma_sym);
	free(expect_sigma);

	/// gpu
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(pi_d));
	checkCudaErrors(cudaFree(alpha_d));
	checkCudaErrors(cudaFree(ones_d));
	checkCudaErrors(cudaFree(beta_d));
	checkCudaErrors(cudaFree(ll_d));
	checkCudaErrors(cudaFree(a_d));
	checkCudaErrors(cudaFree(betaB_d));
	checkCudaErrors(cudaFree(alpha_beta_d));
	checkCudaErrors(cudaFree(gma_d));
	checkCudaErrors(cudaFree(blk_result_d));
	checkCudaErrors(cudaFree(A_alphabetaB_d));
	checkCudaErrors(cudaFree(xi_sum_d));
	checkCudaErrors(cudaFree(expect_A_d));
	checkCudaErrors(cudaFree(gmaT_d));
	checkCudaErrors(cudaFree(gamma_state_sum_d));
	checkCudaErrors(cudaFree(obsT_d));
	checkCudaErrors(cudaFree(gamma_obs_d));
	checkCudaErrors(cudaFree(expect_mu_d));
	checkCudaErrors(cudaFree(obs_d));
	checkCudaErrors(cudaFree(expect_sigma_sym_d));
	checkCudaErrors(cudaFree(expect_sigma_d));
}


/// Calculate beta * B and alpha * beta 
TEST(HMM_BaumWelch, prepare_betaB_alphabeta) 
{
	int current  = 0;
	int previous = N;

	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		b[i]     = 0.15f;
		beta[i]  = 0.2f;
		alpha[i] = 0.3f;
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(b_d,     b,     bytes_nt, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(beta_d,  beta,  bytes_nt, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(alpha_d, alpha, bytes_nt, cudaMemcpyHostToDevice));

	/// kernel
	em_betaB_alphabeta(beta_d,                                              
			           b_d,                                                 
			           betaB_d,                                             
			           alpha_d,                                             
			           alpha_beta_d,                                        
			           N,                                                   
			           current,                                             
			           previous);    

	/// device to host
    checkCudaErrors(cudaMemcpy(betaB,      betaB_d,      bytes_n, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(alpha_beta, alpha_beta_d, bytes_n, cudaMemcpyDeviceToHost));

	/// betaB = 0.03
	/// alpha_beta = 0.06

	/// check results
	int expect = 0;
	for (int i = 0; i < N; i++) {                                             
		if (abs(betaB[i] - 0.03f) >= 1e-5) {
			printf("betaB[%d] = %f\n", i, betaB[i]);
			expect |= 1;		
			break;
		}
	}

	for (int i = 0; i < N; i++) {                                             
		if (abs(alpha_beta[i] - 0.06f) >= 1e-5) {
			printf("alpha_beta[%d] = %f\n", i, alpha_beta[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// update gamma 
TEST(HMM_BaumWelch, update_gamma) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < N; i++) {                                             
		alpha_beta[i] = 0.3f;
	}

	ll[0] = 0.3f;

	int current  = 0;

	/// host to device
    checkCudaErrors(cudaMemcpy(alpha_beta_d, alpha_beta, bytes_n,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ll_d,         ll,         bytes_ll, cudaMemcpyHostToDevice));

	/// kernel
	em_alphabeta_update_gamma (alpha_beta_d, gma_d, ll_d, N, current);

	/// device to host
    checkCudaErrors(cudaMemcpy(gma, gma_d, bytes_nt, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;
	for (int i = 0; i < N; i++) {                                             
		if (abs(gma[current + i] - 1.f) >= 1e-5) {
			printf("gma[%d] = %f\n", current + i, gma[current + i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}



TEST(HMM_BaumWelch, cal_xisum_and_blks) 
{
	int current = 0;
	/// allocate
	allocate_data();

	/// cpu data
	for (int i = 0; i < (N * N); i++) {                                             
		a[i] = 0.15f;
	}

	for (int i = 0; i < (N * T); i++) {                                             
		alpha[i] = 0.3f;
	}

	for (int i = 0; i < N; i++) {                                             
		betaB[i] = 0.1f;
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(a_d,     a,     bytes_nn,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(alpha_d, alpha, bytes_nt,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(betaB_d, betaB, bytes_n,   cudaMemcpyHostToDevice));

	/// kernel
	// A .*  (alpha * betaB')                                               
	em_A_mul_alphabetaB (&alpha_d[current],                                 
			betaB_d,                                           
			bytes_n,                                           
			a_d,                                               
			A_alphabetaB_d,                                    
			blk_result_d,                                      
			N);

	/// device to host
    checkCudaErrors(cudaMemcpy(A_alphabetaB, A_alphabetaB_d, bytes_nn,       cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blk_result,   blk_result_d,   bytes_tileblks, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < N * N; i++) {                                             
		if (abs(A_alphabetaB[i] - 0.0045f) >= 1e-5) {
			printf("A_alphabetaB[%d] = %f\n", i, A_alphabetaB[i]);
			expect |= 1;		
			break;
		}
	}

	/// 1.152
	for(int i = 0; i < tileblks; ++i)                                               
	{                                                                       
		if (abs(blk_result[i] - 1.152f) >= 1e-5) {
			printf("blk_result[%d] = %f\n", i, blk_result[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


TEST(HMM_BaumWelch, update_xisum) 
{
	/// allocate
	allocate_data();

	checkCudaErrors(cudaMemset(xi_sum_d, 0, bytes_nn));

	/// cpu data
	for (int i = 0; i < (N * N); i++) {                                             
		A_alphabetaB[i] = 0.3f;
	}

	float sum = 2.f;

	/// host to device
    checkCudaErrors(cudaMemcpy(A_alphabetaB_d, A_alphabetaB, bytes_nn, cudaMemcpyHostToDevice));

	/// kernel
	em_update_xisum (A_alphabetaB_d,                                        
			xi_sum_d,                                              
			sum,                                                   
			N);

	/// device to host
    checkCudaErrors(cudaMemcpy(xi_sum, xi_sum_d, bytes_nn, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < N * N; i++) {                                             
		if (abs(xi_sum[i] - 0.15f) >= 1e-5) {
			printf("xi_sum[%d] = %f\n", i, xi_sum[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


TEST(HMM_BaumWelch, alphabeta) 
{
	int current = 0;

	/// allocate
	allocate_data();

	/// cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		alpha[i] = 0.2f;
		beta[i]  = 0.3f;
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(alpha_d, alpha, bytes_n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(beta_d,  beta,  bytes_n, cudaMemcpyHostToDevice));

	/// kernel
    em_alphabeta (&beta_d[current],                                             
                  &alpha_d[current],                                            
                  alpha_beta_d,                                                 
                  N);  

	/// device to host
    checkCudaErrors(cudaMemcpy(alpha_beta, alpha_beta_d, bytes_n, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < N; i++) {                                             
		if (abs(alpha_beta[i] - 0.06f) >= 1e-5) {
			printf("alpha_beta[%d] = %f\n", i, alpha_beta[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// update expected A
TEST(HMM_BaumWelch, Update_Expect_A) 
{
	/// allocate
	allocate_data();

	/// cpu data
	for (int i = 0; i < (N * N); i++) {                                             
		xi_sum[i] = 0.2f;
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(xi_sum_d, xi_sum, bytes_nn, cudaMemcpyHostToDevice));

	/// kernel
	em_expect_A (xi_sum_d, expect_A_d, N);  

	/// device to host
    checkCudaErrors(cudaMemcpy(expect_A, expect_A_d, bytes_nn, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	/// 1/64 = 0.015625
	for (int i = 0; i < N * N; i++) {                                             
		if (abs(expect_A[i] - 0.015625f) >= 1e-5) {
			printf("expect_A[%d] = %f\n", i, expect_A[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// transpose gamma
TEST(HMM_BaumWelch, transpose_gma) 
{
	/// allocate
	allocate_data();

	float tmp = 0.f;

	/// cpu data
	for (int row = 0; row < T; row++) {                                             
		tmp = tmp + 1.f;
		for (int col = 0; col < N; col++) {                                             
			gma[row * N + col] = tmp;
		}
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(gma_d, gma, bytes_nt, cudaMemcpyHostToDevice));

	/// kernel
	em_transpose (gma_d, gmaT_d, T, N);   

	/// device to host
    checkCudaErrors(cudaMemcpy(gmaT, gmaT_d, bytes_nt, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	tmp = 0.f;
	float v;
	for (int col = 0; col < T; col++) {                                             
		tmp = tmp + 1.f;
		for (int row = 0; row < N; row++) {                                             
			v = gmaT[row * T + col];
			/// compare
			if (abs(v - tmp) >= 1e-5) {
				printf("gmaT[%d] = %f\n", row * T + col, v);
				expect |= 1;		
				break;
			}
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}



/// sum row on gammaT_d
TEST(HMM_BaumWelch, calc_gamma_state_sum) 
{
	/// allocate
	allocate_data();

	/// cpu data
	for (int row = 0; row < N; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gmaT[row * T + col] = 0.2f;
		}
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(gmaT_d, gmaT, bytes_nt, cudaMemcpyHostToDevice));

	/// kernel
	em_gammastatesum (gmaT_d, gamma_state_sum_d, N, T); 

	/// device to host
    checkCudaErrors(cudaMemcpy(gamma_state_sum, gamma_state_sum_d, bytes_n, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < N; i++) {                                             
		if (abs(gamma_state_sum[i] - 12.8f) >= 1e-5) {
			printf("gamma_state_sum[%d] = %f\n", i, gamma_state_sum[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// gamma obseravtions
TEST(HMM_BaumWelch, gma_observations) 
{
	/// allocate
	allocate_data();

	/// cpu data
	/// obsT : D x T
	for (int row = 0; row < D; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			obsT[row * T + col] = 0.1f;
		}
	}

	/// gmaT : N x T
	for (int row = 0; row < N; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gmaT[row * T + col] = 0.2f;
		}
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(obsT_d, obsT, bytes_dt, cudaMemcpyHostToDevice));

	/// kernel
	em_gammaobs(                                                            
			&gmaT_d[0],                                              
			bytes_t,                                                        
			obsT_d,                                                
			gamma_obs_d,                                                    
			T,                                                              
			D);                 


	/// device to host
    checkCudaErrors(cudaMemcpy(gamma_obs, gamma_obs_d, bytes_dt, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < D * T; i++) {                                             
		if (abs(gamma_obs[i] - 0.02f) >= 1e-5) {
			printf("gamma_obs[%d] = %f\n", i, gamma_obs[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// update expect_mu for each hiddens state
TEST(HMM_BaumWelch, Update_Expect_MU) 
{
	int hs = 0;
	int current = hs * D;

	/// allocate
	allocate_data();

	//-----------------------------------------------------------------------//
	// start calculate gamma_state_sum
	//-----------------------------------------------------------------------//
	/// since the kernel use gamma_state_sum 
	/// we need to generate the data first
	for (int row = 0; row < N; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gmaT[row * T + col] = 0.2f;
		}
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(gmaT_d, gmaT, bytes_nt, cudaMemcpyHostToDevice));

	/// kernel
	em_gammastatesum (gmaT_d, gamma_state_sum_d, N, T); 

	//-----------------------------------------------------------------------//
	// End.     gamma_state_sum  =  12.8
	//-----------------------------------------------------------------------//

	/// gamma_obs: D x T
	for (int row = 0; row < D; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gamma_obs[row * T + col] = 12.8f;
		}
	}

    checkCudaErrors(cudaMemcpy(gamma_obs_d, gamma_obs, bytes_dt, cudaMemcpyHostToDevice));

	/// kernel
	em_expectmu (                                                           
			gamma_obs_d,                                                    
			hs,                                                             
			expect_mu_d,                                                    
			T,                                                              
			D,                                                              
			current);   

	/// device to host
    checkCudaErrors(cudaMemcpy(expect_mu, expect_mu_d, bytes_dn, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < D; i++) {                                             
		if (abs(expect_mu[current + i] - 64.f) >= 1e-5) {
			printf("expect_mu[%d] = %f\n", i, expect_mu[current + i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// prepare expect_sigma 
TEST(HMM_BaumWelch, Expect_Sigma_Dev) 
{
	int hs = 0;
	int current = hs * D;

	/// allocate
	allocate_data();

	//-----------------------------------------------------------------------//
	// (1) prepare obs_d
	//-----------------------------------------------------------------------//
	for (int row = 0; row < T; row++) {                                             
		for (int col = 0; col < D; col++) {                                             
			obs[row * D + col] = 0.1f;
		}
	}
	checkCudaErrors(cudaMemcpy(obs_d, obs, bytes_dt, cudaMemcpyHostToDevice));

	//-----------------------------------------------------------------------//
	// (2) prepare gamma_obs_d 
	//-----------------------------------------------------------------------//
	for (int row = 0; row < D; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gamma_obs[row * T + col] = 12.8f;
		}
	}
	checkCudaErrors(cudaMemcpy(gamma_obs_d, gamma_obs, bytes_dt, cudaMemcpyHostToDevice));

	//-----------------------------------------------------------------------//
	// (3) prepare expect_mu_state and gamma_state_sum 
	//-----------------------------------------------------------------------//
	for (int row = 0; row < N; row++) {                                             
		for (int col = 0; col < T; col++) {                                             
			gmaT[row * T + col] = 0.2f;
		}
	}
    checkCudaErrors(cudaMemcpy(gmaT_d, gmaT, bytes_nt, cudaMemcpyHostToDevice));
	// gamma_state_sum  =  12.8
	em_gammastatesum (gmaT_d, gamma_state_sum_d, N, T); 

	/// kernel
	em_expectmu (                                                           
			gamma_obs_d,                                                    
			hs,                                                             
			expect_mu_d,                                                    
			T,                                                              
			D,                                                              
			current);   
	
	/// 64   for expect_mu_state
	/// 12.8 for gamma_state_sum 

	//-----------------------------------------------------------------------//
	// End of Preparation 
	//-----------------------------------------------------------------------//

	// 81.92 / 12.8 -  64 * 64  = -4089.6 

	/// run kernel
	em_expectsigma_dev(                                                     
			gamma_obs_d,                                                    
			obs_d,                                                 
			hs,                                                             
			expect_sigma_sym_d,                                             
			D,                                                              
			T); 

	/// device to host
    checkCudaErrors(cudaMemcpy(expect_sigma_sym, expect_sigma_sym_d, bytes_dd, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	for (int i = 0; i < D * D; i++) {                                             
		if (abs(expect_sigma_sym[i] + 4089.6f) >= 1e-5) {
			printf("expect_sigma_sym[%d] = %f\n", i, expect_sigma_sym[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


TEST(HMM_BaumWelch, symmetrize_expect_sigma) 
{
	int start = 0;

	/// allocate
	allocate_data();

	for (int row = 0; row < D; row++) {                                             
		for (int col = 0; col < D; col++) {                                             
			expect_sigma_sym[row * D + col] = 1.f;
		}
	}

	//  (0, 16) = 0.2f; 
	//  (0, 15) = 0.2f; 
	expect_sigma_sym[15] = 0.2f;
	expect_sigma_sym[16] = 0.2f;

	checkCudaErrors(cudaMemcpy(expect_sigma_sym_d, expect_sigma_sym, bytes_dd, cudaMemcpyHostToDevice));
	
	/// symmetrize(expect_sigma)                                            
	em_update_expectsigma(                                                  
			expect_sigma_d,                                                 
			expect_sigma_sym_d,                                             
			D,                                                              
			start); 	

	checkCudaErrors(cudaMemcpy(expect_sigma, expect_sigma_d, bytes_ddn, cudaMemcpyDeviceToHost));

	/// check
	int expect = 0;
	// check the (0, 0) = 1.01
	if (abs(expect_sigma[0] - 1.01f ) >= 1e-5) {
		printf("expect_sigma[%d] = %.8f\n", 0, expect_sigma[0]);
		expect |= 1;		
	}
	
	// check the 1st column, compare with 1
	for(int row = 1; row < 15; row++)
	{
		if (abs(expect_sigma[row * D] - 1.f ) >= 1e-5) {
			printf("expect_sigma[%d] = %f\n", row * D, expect_sigma[row * D]);
			expect |= 1;		
		}
	}

	// check (15, 0)
	if (abs(expect_sigma[15 * D] - 0.2f ) >= 1e-5) {
		printf("expect_sigma[%d] = %f\n", 15 * D, expect_sigma[15 * D]);
		expect |= 1;		
	}

	// check (16, 0)
	if (abs(expect_sigma[16 * D] - 0.2f ) >= 1e-5) {
		printf("expect_sigma[%d] = %f\n", 16 * D, expect_sigma[16 * D]);
		expect |= 1;		
	}

	EXPECT_EQ(0, expect);
	
	/// release
	release_data();
}
