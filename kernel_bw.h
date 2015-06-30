#ifndef KERNEL_BW_H
#define KERNEL_BW_H

#include <cuda_runtime.h>      


#ifndef TILE
#define TILE 16	// 2D Kernel Tiling
#endif

#ifndef SIZE 
#define SIZE 4096 
#endif

void em_betaB_alphabeta(
		float *beta_d, 
		float *B_d, 
		float *betaB_d,  
		float *alpha_d,
		float *alpha_beta_d,
		const int N,
		int current,
		int previous);

void em_alphabeta_update_gamma(
		float *alpha_beta_d, 
		float *gamma_d,
		float *ll_d, 
		const int N, 
		unsigned int current);

void em_A_mul_alphabetaB(
		float *alpha_d,
		float *betaB_d,
		size_t bytes_n,
		float *a_d, 
		float *A_alphabetaB_d,
		float *blk_result_d,
		const int N);

void em_update_xisum(
		float *A_alphabetaB_d,
		float *xi_sum_d,
		float sum,
		const int N);

void em_alphabeta(
		float *beta_d, 
		float *alpha_d,
		float *alpha_beta_d,
		const int N);

void em_expect_A(
		float *xi_sum_d,
		float *expect_A_d,
		const int N); 

void em_transpose(
		float *A,
		float *At,
		const int T,
		const int N);


void em_gammastatesum(
		float *gammaT_d,
		float *gamma_state_sum_d,
		const int N,
		const int T);

void em_gammaobs(
		float *gammaT_d,
		size_t bytes_t,
		float *observationsT_d,
		float *gamma_obs_d,
		const int T,
		const int D);

/// cuda stream version
void em_gammaobs_streams(
		float *gammaT_d,
		size_t bytes_t,
		float *observationsT_d,
		float *gamma_obs_d,
		const int T,
		const int D,
		int sid,
		cudaStream_t  *streams);

void em_expectmu(
		float *gamma_obs_d, // D x T
		const int hs,
		float *expect_mu_d, // N x D
		const int T, 
		const int D,
		const uint current);

/// cuda stream version
void em_expectmu_streams(
		float *gamma_obs_d, // D x T
		const int hs,
		float *expect_mu_d, // N x D
		const int T, 
		const int D,
		const uint current,
		int sid,
		cudaStream_t  *streams);


void em_expectsigma_dev(
		float *gamma_obs_d,
		float *observations_d,	
		const int hs,
		float *expect_sigma_sym_d,
		const int D,
		const int T);

/// cuda stream version
void em_expectsigma_dev_streams(
		float *expect_mu_d,
		float *gamma_obs_d,
		float *observations_d,	
		const int hs,
		float *expect_sigma_sym_d,
		const int D,
		const int T,
		int sid,
		cudaStream_t *streams);

void em_update_expectsigma(
		float *expect_sigma_d,	
		float *expect_sigma_sym_d,
		const int D,
		uint start); 	// hs * dd

void em_update_expectsigma_streams(
		float *expect_sigma_d,	
		float *expect_sigma_sym_d,
		const int D,
		uint start,
		int sid,
		cudaStream_t *streams);



#endif
