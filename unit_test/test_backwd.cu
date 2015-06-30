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
int N = 64, T = 64;
float alp = 1.f;                                                                
float bet = 0.f;
size_t bytes_nt  = sizeof(float) * N * T;                                          
size_t bytes_n   = sizeof(float) * N;
size_t bytes_ll  = sizeof(float) * (T + 1);
size_t bytes_nn  = sizeof(float) * N * N;

float *b;    
float *pi;   
float *alpha;
float *ones;
float *beta;
float *ll;
float *a;
float *betaB;

/// gpu memory
float *b_d; 
float *pi_d; 
float *alpha_d; 
float *ones_d; 
float *beta_d;
float *ll_d;
float *a_d;
float *betaB_d;

/// cublas
cublasStatus ret;                                                               
cublasHandle_t handle, handle1;

/// allocate resource
void allocate_data()
{
	/// cpu
	b      = (float *) malloc (bytes_nt);
	pi     = (float *) malloc (bytes_n);
	alpha  = (float *) malloc (bytes_nt);
	ones   = (float *) malloc (bytes_n);
	beta   = (float *) malloc (bytes_nt);
	ll     = (float *) malloc (bytes_ll);
	a      = (float *) malloc (bytes_nn);
	betaB  = (float *) malloc (bytes_n);

	/// gpu
	checkCudaErrors(cudaMalloc((void**)&b_d,     bytes_nt));                        
	checkCudaErrors(cudaMalloc((void**)&pi_d,    bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&alpha_d, bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ones_d,  bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&beta_d,  bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ll_d,    bytes_ll));
	checkCudaErrors(cudaMalloc((void**)&a_d,     bytes_nn));
	checkCudaErrors(cudaMalloc((void**)&betaB_d, bytes_n));  

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


	/// gpu
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(pi_d));
	checkCudaErrors(cudaFree(alpha_d));
	checkCudaErrors(cudaFree(ones_d));
	checkCudaErrors(cudaFree(beta_d));
	checkCudaErrors(cudaFree(ll_d));
	checkCudaErrors(cudaFree(a_d));
	checkCudaErrors(cudaFree(betaB_d));
}



TEST(HMM_Backward, update_beta) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		b[i]    = 0.15f;                                                   
		beta[i] = 0.2f;                                                   
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(b_d,    b,    bytes_nt, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(beta_d, beta, bytes_nt, cudaMemcpyHostToDevice));

	// kernel
	bk_update_beta (&beta_d[0], &b_d[0], betaB_d, N);

	/// device to host
    checkCudaErrors(cudaMemcpy(betaB, betaB_d, bytes_n, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;
	for (int i = 0; i < N; i++) {                                             
		if (abs(betaB[i] - 0.03f) >= 1e-5) {
			printf("betaB[%d] = %f\n", i, betaB[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}



TEST(HMM_Backward, scale_beta) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		beta[i] = 2.f;                                                   
	}

	ll[0] = 4.f; 

	/// host to device
    checkCudaErrors(cudaMemcpy(ll_d,   ll,   bytes_ll, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(beta_d, beta, bytes_nt, cudaMemcpyHostToDevice));

	// kernel
	bk_scaling (N, &beta_d[0], ll_d); 

	/// device to host
    checkCudaErrors(cudaMemcpy(beta, beta_d, bytes_nt, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;
	for (int i = 0; i < N; i++) {                                             
		if (abs(beta[i] - 0.5f) >= 1e-5) {
			printf("beta[%d] = %f\n", i, beta[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}
