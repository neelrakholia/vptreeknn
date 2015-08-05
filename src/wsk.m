function d = wsk(X, x, sigma, sublen)
%--------------------------------------------------------------------------
% WSK evaluates the word sequence kernel distance between pairs of points 
% in X and x
%   Input 
%       X - Array of points
%       x - Array of points
%       sigma - Lambda value desired
%       sublen - word sequence length
%
%   Output 
%       d - evaluated kernel value
%--------------------------------------------------------------------------

% get lengths of arrays
n = length(x);
N = length(X);

% initialize output array
d = zeros(N,n);

% loop over the matrix and compute lengths
for i = 1:N
    for j = 1:n
        d(i,j) = - 2 * ssk_dyn(X{i},x{j},sublen,sigma)/...
            sqrt(ssk_dyn(X{i},X{i},sublen,sigma)*...
            ssk_dyn(x{j},x{j},sublen,sigma));
    end
end