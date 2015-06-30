#ifndef CONSTANTS_BW_H
#define CONSTANTS_BW_H

#ifndef TILE
	#define TILE 16	// 2D Kernel Tiling
#endif

// size = hidden state number
#ifndef SIZE 
	#define SIZE 4096 
#endif

// cache global memory
// 64 KB on kepler
__constant__ float ConstA[SIZE];	// alpha_d
__constant__ float ConstB[SIZE];	// betaB_d

__constant__ float gamma_state_sumC[SIZE];
__constant__ float bufferT[64];
__constant__ float expect_mu_state[64]; // D



#endif
