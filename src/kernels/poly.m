function d = poly(X, x, h, c, p)
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

d = (X'*x/h + c).^p;

end