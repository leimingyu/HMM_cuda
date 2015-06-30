#ifndef KERNEL_FORWARD_H
#define KERNEL_FORWARD_H

void fwd_init_alpha(float *b_d, 
					float *pi_d, 
					int N, 
					float *alpha_d, 
					float *ones_d,
					float *beta_d);

void fwd_scaling(const int N, 
                 float *alpha_d, 
	             float *ll_d, 
	             int t);

void fwd_calc_alpha(const int N, 
                    float *alpha_d, 
	                float *b_d);

void fwd_sum_ll(const int T, 
		        float *ll_d);
#endif
