function [ d ] = cosine_kernel_dist( x, y )

% self interaction is always 1 because the kernel 
d = 2*ones(size(x,2), size(y,2)) - cosine_kernel(x, y);

end

