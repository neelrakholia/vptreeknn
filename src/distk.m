function d = distk(X, x, sigma)
%--------------------------------------------------------------------------
% DISTK Computes the distance for kernels
%   Input 
%       X - Array of points
%       x - Array of points
%       sigma - Kernel bandwidth parameter
%
%   Output 
%       d - evaluated kernel value
%--------------------------------------------------------------------------
% select kernel type
kerneltype = 'wsk';

% Add switch here to change kernel type used
if(strcmp(kerneltype,'rbf'))
    d = rbf(X, x, sigma);
else
    d = wsk(X, x, sigma, 2);
end

end