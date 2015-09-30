function d = rbf(X, x, sigma)
%--------------------------------------------------------------------------
% RBF Evaluates pairwise RBF kernel distances between points in 2 arrays 
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
d = repmat(x2,N,1).^10 +repmat(X2',1,n).^10 -2*dotProd.^10;

% evaluate kernel
d = sqrt(d);

end