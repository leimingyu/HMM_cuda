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

/// gpu memory
float *b_d; 
float *pi_d; 
float *alpha_d; 
float *ones_d; 
float *beta_d;
float *ll_d;
float *a_d;

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

	/// gpu
	checkCudaErrors(cudaMalloc((void**)&b_d,     bytes_nt));                        
	checkCudaErrors(cudaMalloc((void**)&pi_d,    bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&alpha_d, bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ones_d,  bytes_n));     
	checkCudaErrors(cudaMalloc((void**)&beta_d,  bytes_nt));
	checkCudaErrors(cudaMalloc((void**)&ll_d,    bytes_ll));
	checkCudaErrors(cudaMalloc((void**)&a_d,     bytes_nn));

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


	/// gpu
	checkCudaErrors(cudaFree(b_d));
	checkCudaErrors(cudaFree(pi_d));
	checkCudaErrors(cudaFree(alpha_d));
	checkCudaErrors(cudaFree(ones_d));
	checkCudaErrors(cudaFree(beta_d));
	checkCudaErrors(cudaFree(ll_d));
	checkCudaErrors(cudaFree(a_d));
}



/// forward algo: initialize step
TEST(HMM_Forward, intialization) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		b[i] = 0.3f;                                                   
	}

	for (int i = 0; i < N; i++) {                                                   
		pi[i] = 0.3f;                                                  
	}     

	/// host to device
    checkCudaErrors(cudaMemcpy(b_d,  b,  bytes_nt, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pi_d, pi, bytes_n,  cudaMemcpyHostToDevice));


	/// alpha = b * pi
	/// initialize ones_d for cublas
	/// initialize beta_d
	fwd_init_alpha (b_d,                                                        
			pi_d,                                                       
			N,                                                          
			&alpha_d[0],		// the first N samples
			ones_d,                                                     
			&beta_d[(T-1)*N]);  

	/// device to host
    checkCudaErrors(cudaMemcpy(alpha, alpha_d, bytes_nt, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(ones, ones_d, bytes_n, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(beta, beta_d, bytes_nt, cudaMemcpyDeviceToHost));

	/// check results
	int expect = 0;

	/// check the first N samples of alpha_d
	for (int i = 0; i < N; i++) {                                             
		/// if the difference is larger than 0.00001, trigger the alarm 
		if (abs(alpha[i] - 0.09f) >= 1e-5) {
			expect |= 1;		
			break;
		}
	}

	/// check ones_d
	for (int i = 0; i < N; i++) {                                             
		/// if the difference is larger than 0.00001, trigger the alarm 
		if (abs(ones[i] - 1.f) >= 1e-5) {
			expect |= 1;		
			break;
		}
	}

	/// check beta_d
	for (int i = (T-1) * N; i < T * N; i++) {                                             
		/// if the difference is larger than 0.00001, trigger the alarm 
		if (abs(beta[i] - 1.f) >= 1e-5) {
			expect |= 1;		
			break;
		}
	}


	EXPECT_EQ(0, expect);

	/// release
	release_data();
}



// test cublas_sdot
TEST(HMM_Forward, sum_alpha) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		alpha[i] = 0.3f;                                                   
	}

	for (int i = 0; i < N; i++) {                                                   
		ones[i] = 1.0f;                                                  
	}     

	/// host to device
    checkCudaErrors(cudaMemcpy(alpha_d,  alpha, bytes_nt, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ones_d,   ones,  bytes_n,  cudaMemcpyHostToDevice));

	/// sum(alpha)                                                               
	ret = cublasSdot(handle, N, &alpha_d[0], 1, ones_d, 1, &ll_d[0]);           
	if (ret != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "ERROR: Sdot execution error. This is line %d.\n", __LINE__);
		exit(EXIT_FAILURE);                                                     
	}    

	/// device to host
    checkCudaErrors(cudaMemcpy(ll, ll_d, bytes_ll, cudaMemcpyDeviceToHost));
	
	/// check results
	int expect = 0;

	if (abs(ll[0] - 19.2f) >= 1e-5) {
		expect = 1;		
	}

	// printf("ll[0] = %f\n", ll[0]);

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


// parallel division 
TEST(HMM_Forward, scaling_alpha) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		alpha[i] = 0.3f;                                                   
	}

	ll[0] = 2.0f;                                                  

	/// host to device
    checkCudaErrors(cudaMemcpy(alpha_d,  alpha, bytes_nt,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ll_d,     ll,    bytes_ll,  cudaMemcpyHostToDevice));

	/// kernel
	fwd_scaling (N , &alpha_d[0], ll_d, 0);

	/// device to host
    checkCudaErrors(cudaMemcpy(alpha, alpha_d, bytes_nt, cudaMemcpyDeviceToHost));
	
	/// check results
	int expect = 0;

	for (int i = 0; i < N; i++) {                                             
		if (abs(alpha[i] - 0.15f) >= 1e-5) {
			printf("alpha[%d] = %f\n", i, alpha[i]);
			expect |= 1;		
			break;
		}
	}
	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// test a' * alpha
TEST(HMM_Forward, aT_mul_alpha) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * N); i++) {                                             
		a[i] = 0.4f;                                                   
	}

	/// 1st column to 0.2f
	for (int row = 0; row < N ; row++) {                                             
		a[row * N] = 0.2f;                                                   
	}

	for (int i = 0; i < N; i++) {                                                   
		alpha[i] = 1.0f;                                                  
	}     

	/// host to device
    checkCudaErrors(cudaMemcpy(a_d,     a,     bytes_nn, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(alpha_d, alpha, bytes_nt,  cudaMemcpyHostToDevice));

	// a' * alpha                                                           
	// auto transposed due to the column major thing                        
	ret = cublasSgemv(handle1, CUBLAS_OP_N,                                 
			N, N,                                                           
			&alp,                                                           
			a_d, N,                                                         
			&alpha_d[0], 1,                                          
			&bet,                                                           
			&alpha_d[N], 1);                                          

	if (ret != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "ERROR: Sgemv execution error. This is line %d.\n", __LINE__);
		exit(EXIT_FAILURE);                                                 
	}  

	/// device to host
    checkCudaErrors(cudaMemcpy(alpha, alpha_d, bytes_nt, cudaMemcpyDeviceToHost));
	
	/// check results
	int expect = 0;

	// alpha[N] = 12.8; the others should be 25.6;
	for (int i = 0; i < N; i++) {                                             
		if(i == 0)
		{
			if (abs(alpha[N + i] - 12.8f) >= 1e-5) {
				printf("alpha[%d] = %f\n", N + i, alpha[N + i]);
				expect |= 1;		
				break;
			}
		}
		else
		{
			if (abs(alpha[N + i] - 25.6f) >= 1e-5) {
				printf("alpha[%d] = %f\n", N + i, alpha[N + i]);
				expect |= 1;		
				break;
			}
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}



/// b * (a' * alpha) 
TEST(HMM_Forward, update_alpha) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < (N * T); i++) {                                             
		alpha[i] = 0.3f;                                                   
		b[i]     = 0.2f;                                                   
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(alpha_d, alpha, bytes_nt,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(b_d,     b,     bytes_nt,  cudaMemcpyHostToDevice));

	/// kernel
	/// b * (a' * alpha)                                                     
	fwd_calc_alpha (N, &alpha_d[0], &b_d[0]);   

	/// device to host
    checkCudaErrors(cudaMemcpy(alpha, alpha_d, bytes_nt, cudaMemcpyDeviceToHost));
	
	/// check results
	int expect = 0;

	for (int i = 0; i < N; i++) {                                             
		if (abs(alpha[i] - 0.06f) >= 1e-5) {
			printf("alpha[%d] = %f\n", i, alpha[i]);
			expect |= 1;		
			break;
		}
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


/// sum up loglikelihood
TEST(HMM_Forward, sum_log_likelihood) 
{
	/// allocate
	allocate_data();

	/// configure cpu data
	for (int i = 0; i < T; i++) {                                             
		ll[i] = 10.f;                                                   
	}

	/// host to device
    checkCudaErrors(cudaMemcpy(ll_d, ll, bytes_ll,  cudaMemcpyHostToDevice));

	/// kernel
	fwd_sum_ll (T, ll_d);  

	/// device to host
    checkCudaErrors(cudaMemcpy(ll, ll_d, bytes_ll, cudaMemcpyDeviceToHost));
	
	/// check results
	int expect = 0;

	if (abs(ll[T] - 64.f) >= 1e-5) {
		printf("log10(ll)= %f\n", ll[T]);
		expect = 1;		
	}

	EXPECT_EQ(0, expect);

	/// release
	release_data();
}


