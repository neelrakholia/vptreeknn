function d = rbf_dist(X, x, sigma)
%--------------------------------------------------------------------------
% RBF Evaluates pairwise RBF kernel distances between points in 2 arrays,
% returns the distance imposed by the kernel
%   Input 
%       X - Array of points
%       x - Array of points
%       sigma - Kernel bandwidth parameter
%
%   Output 
%       d - evaluated kernel value
%--------------------------------------------------------------------------
% evaluate euclidean part
[~,N] = size(X);
[~,n] = size(x);
X2 = sum(X.^2,1);
x2 = sum(x.^2,1);
dotProd = X'*x;
d = repmat(x2,N,1)+repmat(X2',1,n)-2*dotProd;

% evaluate kernel
d = exp(-d/(2*sigma^2));

d = sqrt(-2*d + 2);
% d = d./(d + 1);
end