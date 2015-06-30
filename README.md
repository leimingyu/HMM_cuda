# HMM_cuda
Hidden Markov Model for Isolated Word Recognition on GPU
```
Author:  Leiming Yu
         ylm@ece.neu.edu
         Northeastern University, Boston, MA, USA
```

It is implemented in the following paper. Please refer to it if you want to use. Thanks.

```
@inproceedings{yu2014gpu,
  title={GPU-accelerated HMM for Speech Recognition},
  author={Yu, Leiming and Ukidave, Yash and Kaeli, David},
  booktitle={Parallel Processing Workshops (ICCPW), 2014 43rd International Conference on},
  pages={395--402},
  year={2014},
  organization={IEEE}
}
```

Usage:
```
To compile:
	$cmake .
	$make
To verify each module:
	$make test
	or run each test separately:
	$./test_forword
	$./test_backwd
	$./test_bw
To run program:
	$./hmmsr number_of_hidden_states number_of_streams
	e.g., $./hmmsr 256 4
To turn on/off hyper-q:
	In hmmsr.cu, change #define HQ 1 or 0.
```
