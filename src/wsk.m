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
X_self = zeros(N,1);
x_self = zeros(n,1);

% pre-compute self kernel values
for k = 1:n
    x_self(k) = ssk_dyn_mex(x{k},length(x{k}),x{k},length(x{k}),...
        sublen,sigma);
end

for k = 1:N
    X_self(k) = ssk_dyn_mex(X{k},length(X{k}),X{k},length(X{k}),...
        sublen,sigma);
end

% loop over the matrix and compute lengths
for i = 1:N
    for j = 1:n
        d(i,j) = - 2 * ssk_dyn_mex(X{k},length(X{i}),x{k},...
            length(x{j}),sublen,sigma)/...
            sqrt(X_self(i)*x_self(j));
    end
end