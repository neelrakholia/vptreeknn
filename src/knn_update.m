function [ dists_out, inds_out ] = knn_update( dists_in, inds_in, k )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%     [inds_in, dist_inds, ~] = unique(inds_in(i,:));
%     dists_in(i,:) = dists_in(i,:)(dist_inds);

% changed this to make it max similarity search now
[sort_vals, sort_inds] = sort(dists_in, 2, 'descend');

% for i = 1:k
%     assert(sort_inds(i) == i);
% end

% if we don't have enough
% if (numel(sort_inds) < k)
%   
%     num_needed = k - numel(sort_inds);
%     dists_out = [dists_in, realmax * ones(1, num_needed)];
%     inds_out = [inds_in, zeros(1, num_needed)];
%     
% else
    
    dists_out = sort_vals(:,1:k);
% TODO: make less stupid
    inds_out = zeros(size(dists_in,1),k);
    for i=1:size(dists_in,1)
%         unique_inds = unique(inds_in(i,sort_inds(i,:)), 'stable');
        inds_out(i,:) = inds_in(i,sort_inds(i,1:k));
    end

% end


% D=distance(query,reference);
% [Ds,in]=sort(D,2, 'ascend');
% 
% idk   = in(:,1:k_neighbors);
% dists = D;




end

