
#include <matrix.h>
#include <math.h>
#include <mex.h>

mxArray *mxGetCell(const mxArray *pm, mwIndex index);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  
  // inputs are two cell arrays
  // one contains the bin information (size, count) for each bin
  // the other is the index information to re-build the tree structure
  
  // number of data points
  int num_rows;
  int num_cols;
  
  int size[2];
  size[0] = num_rows;
  size[1] = 2;
  
  mxArray* row_bin_data = mxCreateCellArray(2, 2*num_rows);
  
  
  
}





