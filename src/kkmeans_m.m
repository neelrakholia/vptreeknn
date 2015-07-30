% NOTE: THIS FILE IS NO LONGER USED IN OUR METHOD. FOR INFORMATIONAL
% PURPOSES ONLY

function [label, medoid] = kkmeans_m(points, kernelf, k, n)
%--------------------------------------------------------------------------
% KKMEANS_M Runs kernel k-medoid and outputs labels and centroidss
%   Input 
%       points - data to be clustered
%       kernelf - kernel function
%       k - number of nearest neighbors that have to be computed
%       n - total number of points
%
%   Output 
%       label - whether points belong to the left or the right node
%       medoid - cluster centers of each node
%--------------------------------------------------------------------------
% get distance matrix
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = kernelf(points(:, i), points(:, j));
    end
end

% run kernel k-medoid
cluster = kkmedoid(points', k, K);
label = cluster.label;
medoid = cluster.medoids;

end