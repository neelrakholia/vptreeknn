function [ accuracy, dist_evals ] = klsh( queries, references, k, b, B, M, p, kernelfun)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[~, num_queries] = size(queries);
[~, num_references] = size(references);

% form the estimated covariance matrix
references = references(:, randperm(num_references));

Kp = kernelfun(references(:,1:p), references(:,1:p));

% create the hash matrix
[~, W] = createHashTable(Kp, p, b);

% now, hash the remaining points
H_ref = kernelfun(references, references(:,1:p)) * W > 0;
H_query = kernelfun(queries, references(:,1:p)) * W > 0;

neighbor_dists = ones(num_queries, k) * realmax;
neighbor_inds = zeros(num_queries, k);

% compute actual nn for accuracy purposes
% piece = min(100, num_queries);
% actual_nn = zeros(num_queries,k);
% actual_nn(1:num_queries/piece, :) = kknn(queries,1:num_references, references(:,1:num_queries/piece),0.22,k,num_references);
% for i = 2:piece
%     actual_nn((i - 1)*(num_queries/piece) + 1:i*num_queries/piece,:) = ...
%         kknn(queries,1:num_references,references(:,(i - 1)*(num_queries/piece) + 1:i*num_queries/piece),0.22,k,num_references);
% end
actual_nn = kknn(references, 1:num_references, queries, 0.22, k, num_references);


% count distance computations
total_comps = 0;


% now, loop over permutations
for i = 1:M
  
    perm = randperm(b);

    Compact_ref = compactbit(H_ref(:,perm));
    [sorted_ref, sort_inds] = sort(Compact_ref);
    
    Compact_query = compactbit(H_query(:,perm));
    

    % now, get the distances between the queries and their candidates
    dists = 10^10 * ones(num_queries, 2*B);
    
    for j = 1:num_queries

        
         lower_inds = find(Compact_query(j,:) < sorted_ref, B);
         upper_inds = find(Compact_query(j,:) >= sorted_ref, B, 'last');
         
         inds = sort_inds(union(lower_inds, upper_inds))';
         
         % if we don't find 2*B elements
         if(length(inds) ~= 2*B)
             lower_inds = find(Compact_query(j,:) <= sorted_ref, 2*B);
             inds = sort_inds(lower_inds)';
         end
         if(length(inds) ~= 2*B)
             upper_inds = find(Compact_query(j,:) >= sorted_ref, 2*B, 'last');
             inds = sort_inds(upper_inds)';
         end
%         inds = binarySearch(sorted_ref, Compact_query(j,:));
%         inds = unique(sort_inds(inds));     
        dists(j,1:numel(inds)) = kernelfun(queries(:,j), references(:, inds))';  
        total_comps = total_comps + numel(inds);
        
        [neighbor_dists(j,:), neighbor_inds(j,:)] = knn_update([neighbor_dists(j,:), dists(j,:)], [neighbor_inds(j,:), inds], k);
        
    end
    
    % estimate the accuracy in each iteration
    sum = 0;
    for j = 1:num_queries
        sum = sum + length(intersect(actual_nn(j,:), neighbor_inds(j,:)));
    end
    acc = sum/(num_queries*k);

    fprintf('Iteration %d: Accuracy %g with %g dist evals.\n', i, acc, (p*p + total_comps)/(num_queries*num_references));
    
end

accuracy = acc;
dist_evals = (p*p + total_comps)/(num_queries*num_references);

end

