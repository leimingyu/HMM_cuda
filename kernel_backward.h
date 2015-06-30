#ifndef KERNEL_BACKWARD_H
#define KERNEL_BACKWARD_H

void bk_update_beta(float *beta_d, 
                    float *B_d, 
                    float *betaB_d, 
                    const int N);

void bk_scaling(const int N, 
                float *beta_d, 
                float *ll_d);

#endif
