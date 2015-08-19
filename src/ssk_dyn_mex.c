/*
 * ssk_dyn_mex.c 
 *
 * Computes ssk using dynamic programming approach
 *
 * This is a MEX-file for MATLAB.
*/

// libraries to include

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "ssk_dyn.h"
#include "mex.h"
#include "matrix.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    // Check for appropriate input size
    if(nrhs != 6) {
    mexErrMsgIdAndTxt("MyToolbox:ssk_dyn_mex:nrhs",
                      "Six inputs required.");
    }

    if(nlhs != 1) {
    mexErrMsgIdAndTxt("MyToolbox:ssk_dyn_mex:nlhs",
                      "One output required.");
    }
    
    // input variables
    char *s1;
    size_t len1;
    char *s2;
    size_t len2;
    size_t sublen;
    double lambda;
    
    // get input variables
    len1 = mxGetScalar(prhs[1]);
    len2 = mxGetScalar(prhs[3]);
    
    // allocate arrays
    char **array1 = (char **)mxMalloc(sizeof(char *)*len1);
    char **array2 = (char **)mxMalloc(sizeof(char *)*len2);
    
    // fill arrays
    int i;
    for(i = 0; i < len1; i++) {
     
        array1[i] = mxArrayToString(mxGetCell (prhs[0], i));
        
    }
    
    for(i = 0; i < len2; i++) {
     
        array2[i] = mxArrayToString(mxGetCell (prhs[2], i));
        
    }
    
    // get all the other inputs
    sublen = mxGetScalar(prhs[4]);
    lambda = mxGetScalar(prhs[5]);
    
    // generate output
    plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
    double *b = mxGetPr(plhs[0]);
    *b = ssk_dyn(array1, len1, array2, len2, sublen, lambda);
     
    // free heap space
    mxFree(array1);
    mxFree(array2);
}

double ssk_dyn(char *s1[], size_t len1, char *s2[], size_t len2, 
        size_t sublen, double lambda) {
//-------------------------------------------------------------------------
//SSK_DYN Computes ssk using dynamic programming approach
//   Input
//       s1 - string 1
//       len1 - length of string1
//       s2 - string 2
//       len2 - length of string2
//       sublen - subsequence length
//       lambda - tuning parameter
//
//   Output
//       k - kernel value
//-------------------------------------------------------------------------

    // create matrix of zeros
    double kp[len1 + 1][len2 + 1][sublen + 1]; 
    memset(kp, 0, (len1 + 1)*(len2 + 1)*(sublen + 1)*sizeof(double));

    // initialize third column to 1
    int i,j,k;
    for(i = 0; i < len1 + 1; i++) {

        for(j = 0; j < len2 + 1; j++) {

            kp[i][j][0] = 1;

        }   

    }   

    int s_ind,t_ind;
    // loop for subsequence length
    for(i = 1; i < sublen + 1; i++) {

        // loop over first string length
        for(s_ind = 1; s_ind < len1 + 1; s_ind++) {

            // update kp by multiplying by lambda
            for(k = 0; k < len2 + 1; k++) {

                kp[s_ind][k][i] += lambda * kp[s_ind - 1][k][i];

            }   

            // loop over second substring
            for(t_ind = 1; t_ind < len2 + 1; t_ind++) {

                // indices in s2 such that s1(s_ind) matches entry of s2
                int s2_inds[len2 + 1], x, y;
                memset(s2_inds, 0, (len2 + 1)*sizeof(int));
                y = 0;
                for(x = 0; x < t_ind; x++) {

                    if(strcmp(s2[x],s1[s_ind - 1]) == 0) {

                        s2_inds[y] = x;
                        y++;

                    }

                }

                // compute length of s2_inds
                int len = y;
                double sum = 0;

                // sum over prev K' to get new K'
                for(x = 0; x < len; x++) {

                    sum += kp[s_ind - 1][s2_inds[x]][i - 1] *
                        pow(lambda,(t_ind - s2_inds[x] + 1));

                }

                // update kp
                kp[s_ind][t_ind][i] += sum;

            }

        }

    }

    // calculate K
    double K[len1 + 1];
    memset(K, 0, (len1 + 1)*sizeof(double));

    // loop over first string length to get kernel value
    for(s_ind = 1; s_ind < len1 + 1; s_ind++) {


        int s2_inds[len2 + 1], x, y;
        memset(s2_inds, 0, (len2 + 1)*sizeof(int));
        y = 0;
        for(x = 0; x < t_ind - 1; x++) {

            if(strcmp(s2[x],s1[s_ind - 1]) == 0) {

                s2_inds[y] = x;
                y++;

            }
                                                                                                                                                                                       
        }

        // compute length of s2_inds
        int len = y;
        double sum = 0;

        // sum over prev K' to get new K'
        for(x = 0; x < len; x++) {

            sum += kp[s_ind - 1][s2_inds[x]][sublen - 1];

        }

        K[s_ind] = K[s_ind - 1] + sum * pow(lambda,2);

    }

   return K[len1];

}
