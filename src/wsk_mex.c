/*
 * wsk_mex.c 
 *
 * Computes ssk using dynamic programming approach for all the data points 
 * in the given matrices
 *
 * This is a MEX-file for MATLAB.
*/

// libraries to include

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "ssk_dyn.h"

typedef uint16_t char16_t;

#include "mex.h"
#include "matrix.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    // Check for appropriate input size
    if(nrhs != 4) {
    mexErrMsgIdAndTxt("MyToolbox:wsk_mex:nrhs",
                      "Four inputs required.");
    }
    
    // define arrays for input and output
    mxArray *normInputs[6];
    mxArray *normOutputs[1];
    
    // read sublen and lambda
    mwSize dims[1];
    dims[0] = 1;
    mxArray *sublen = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *lambda = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(sublen) = mxGetScalar(prhs[3]);
    normInputs[4] = sublen;
    *mxGetPr(lambda) = mxGetScalar(prhs[2]);
    normInputs[5] = lambda;
    
    // get size of cell arrays
    const int *n_dim = mxGetDimensions(prhs[1]);
    const int *N_dim = mxGetDimensions(prhs[0]);
    int n = n_dim[1];
    int N = N_dim[1];
    
    // define arrays to store self kernel values
    double X_self[N];
    double x_self[n];
    memset(X_self, 0, N*sizeof(double));
    memset(x_self, 0, n*sizeof(double));
    
    // compute self kernel values
    int k;
    
    // Define other arrays for storing intermediate values
    mxArray *len1 = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    mxArray *len2 = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
    
    
    // mexPrintf("loop over n\n");
    for(k = 0; k < n; k++) {
        
      // mexPrintf("loop over n, iter: %d\n", k);
        
        // Assign values to input for ssk_dyn_mex
        normInputs[0] = mxGetCell(prhs[1],k);
        const int *dimen = mxGetDimensions(normInputs[0]);
        *mxGetPr(len1) = dimen[1];
        normInputs[1] = len1;
        normInputs[2] = mxGetCell(prhs[1],k);
        const int *dimen2 = mxGetDimensions(normInputs[2]);
        *mxGetPr(len2) = dimen2[1];
        normInputs[3] = len2;
        // mexPrintf("Calling ssk_dyn_mex\n");
      
        mexCallMATLAB(1,normOutputs,6,normInputs,"ssk_dyn_mex");
        
        // assign output value
        x_self[k] = *mxGetPr(normOutputs[0]);
        
    }

    // mexPrintf("loop over N\n");
    for(k = 0; k < N; k++) {
        
        // Assign values to input for ssk_dyn_mex
        normInputs[0] = mxGetCell(prhs[0],k);
        const int *dimen = mxGetDimensions(normInputs[0]);
        *mxGetPr(len1) = dimen[1];
        normInputs[1] = len1;
        normInputs[2] = mxGetCell(prhs[0],k);
        const int *dimen2 = mxGetDimensions(normInputs[2]);
        *mxGetPr(len2) = dimen2[1];
        normInputs[3] = len2;
        mexCallMATLAB(1,normOutputs,6,normInputs,"ssk_dyn_mex");
        
        // assign output value
        X_self[k] = *mxGetPr(normOutputs[0]);
        
    }
      
    // compute all kernel values
    
    // initialize 2-d array
    
    // don't need d, just write to outputptr directly
    // double d[N][n];
    // memset(d, 0, N*n*sizeof(double));
    plhs[0] = mxCreateDoubleMatrix(N, n, mxREAL);
    // get pointer to output array
    double *outputptr = mxGetPr(plhs[0]);
    
    // mexPrintf("Nested loop \n");
    

    // loop over all matrix and compute distances 
    int i,j,count;
    count = 0;

    // Need to output in row-order, so iterate over columns first 
    for(j = 0; j < n; j++) {
            
      for(i = 0; i < N; i++) {
       
            // Assign values to input for ssk_dyn_mex
            normInputs[0] = mxGetCell(prhs[0],i);
            const int *dimen = mxGetDimensions(normInputs[0]);
            *mxGetPr(len1) = dimen[1];
            normInputs[1] = len1;
            normInputs[2] = mxGetCell(prhs[1],j);
            const int *dimen2 = mxGetDimensions(normInputs[2]);
            *mxGetPr(len2) = dimen2[1];
            normInputs[3] = len2;
            mexCallMATLAB(1,normOutputs,6,normInputs,"ssk_dyn_mex");
            
            // assign output value
            double kvalue = *mxGetPr(normOutputs[0]);

            // compute kernel distance
            outputptr[count++] = kvalue;
                    
        }
        
    }
    
    // free memory
    mxDestroyArray(len1);
    mxDestroyArray(len2);
    
    // free memory for all arrays
    mxDestroyArray(normInputs[5]);
    mxDestroyArray(normInputs[4]);        
    //mxDestroyArray(normInputs[3]);
    //mxDestroyArray(normInputs[1]);
    mxDestroyArray(normOutputs[0]);
    
    //mxDestroyArray(sublen);
    //mxDestroyArray(lambda);
        
}