function d = poly_dist(X, x, h, c, p)
%--------------------------------------------------------------------------
% RBF Evaluates pairwise RBF kernel distances between points in 2 arrays 
%   Input 
%       X - Array of points
%       x - Array of points
%       h -- scale
%       c -- low-order offset
%       p -- power
%
%       Computes (X'x/h + c)^p
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

d = (repmat(x2,N,1)/h + c).^p + (repmat(X2',1,n)/h + c).^p;

% evaluate kernel
% need the minus sign because we always search for the max now
d = sqrt(d - 2 * (dotProd/h + c).^p);

end