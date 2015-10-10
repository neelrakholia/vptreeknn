function [ d ] = hyptan_kernel_dist( x, y, h, c )

[~,N] = size(x);
[~,n] = size(y);
x2 = sum(x.^2,1);
y2 = sum(y.^2,1);
dotProd = x'*y;

d = tanh(repmat(y2,N,1)/h + c) + tanh(repmat(x2',1,n)/h + c) - 2 * tanh(dotProd/h + c * ones(N,n));

end

