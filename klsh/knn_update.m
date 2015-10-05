function [ dists_out, inds_out ] = knn_update( dists_in, inds_in, k )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[inds_in, dist_inds, ~] = unique(inds_in);
dists_in = dists_in(dist_inds);

[~, sort_inds] = sort(dists_in);

% if we don't have enough
if (numel(sort_inds) < k)
  
    num_needed = k - numel(sort_inds);
    dists_out = [dists_in, realmax * ones(1, num_needed)];
    inds_out = [inds_in, zeros(1, num_needed)];
    
else
    
    dists_out = dists_in(sort_inds(1:k));
    inds_out = inds_in(sort_inds(1:k));

end


end

