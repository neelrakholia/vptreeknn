% NOTE: THIS FILE IS NO LONGER USED IN OUR METHOD. FOR INFORMATIONAL
% PURPOSES ONLY

function label = kkmeans_ap(points, kernelf, k, n)
%--------------------------------------------------------------------------
% KKMEANS_AP Runs kernel k-means
%   Input 
%       points - data to be clustered
%       kernelf - kernel function
%       k - number of nearest neighbors that have to be computed
%       n - total number of points
%
%   Output 
%       label - whether points belong to the left or the right node
%--------------------------------------------------------------------------
% set maximum number of iterations
max_iter = 100;

% the number of sample points that we take
m = floor(n/4);

% sample m data points
perm = randperm(n);
indices = perm(1:m);

% get kernel matrix
K = zeros(m, n);
for i = 1:m
    for j = 1:n
        K(i, j) = kernelf(points(:, indices(i)), points(:, j));
    end
end

% run approximate kernel k-means algorithm
label = approx_kkmeans(K, k, max_iter, indices);
end

