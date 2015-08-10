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

% pre-compute self kernel values
x_self = cellfun(@(y) ssk_dyn(y,y,sublen,sigma),x);
X_self = cellfun(@(y) ssk_dyn(y,y,sublen,sigma),X);

% loop over the matrix and compute lengths
for i = 1:N
    for j = 1:n
        d(i,j) = - 2 * ssk_dyn(X{i},x{j},sublen,sigma)/...
            sqrt(X_self(i)*x_self(j));
        d(i,j)
    end
end