function [ rank_acc, dist_acc, dist_evals ] = klsh( queries, references, k, b, B, M, p, t, kernelfun, actual_dists, actual_nn)

[~, num_queries] = size(queries);
[~, num_references] = size(references);

% form the estimated covariance matrix
perm_inds = randperm(num_references);
perm_inds = perm_inds(1:p);

fprintf('\nCreating hash functions\n');
Kp = kernelfun(references(:,perm_inds), references(:,perm_inds));

% create the hash matrix
[~, W] = createHashTable(Kp, t, b);

% now, hash the remaining points
H_ref = kernelfun(references, references(:,perm_inds)) * W > 0;
H_query = kernelfun(queries, references(:,perm_inds)) * W > 0;

fprintf('\nFinished hash functions\n');

neighbor_dists = ones(num_queries, k) * -inf;
neighbor_inds = zeros(num_queries, k);

hash_evals = p*p + p * num_references;

% count distance computations
search_evals = p*num_queries;

% now, loop over permutations
for i = 1:M
  
    perm = randperm(b);

    Compact_ref = compactbit(H_ref(:,perm));
    [sorted_ref, sort_inds] = sort(Compact_ref);
    
    Compact_query = compactbit(H_query(:,perm));
    

    % now, get the distances between the queries and their candidates
    dists = -inf * ones(num_queries, 2*B);
    
    for j = 1:num_queries

         lower_inds = find(Compact_query(j,:) < sorted_ref, B);
         upper_inds = find(Compact_query(j,:) >= sorted_ref, B, 'last');
         
         inds = sort_inds(union(lower_inds, upper_inds))';
%         inds = binarySearch(sorted_ref, Compact_query(j,:));
%         inds = unique(sort_inds(inds));

        inds = setdiff(inds, neighbor_inds(j,:));
        
        dists(j,1:numel(inds)) = kernelfun(queries(:,j), references(:, inds))';  
        
        search_evals = search_evals + numel(inds);
        
        [neighbor_dists(j,:), neighbor_inds(j,:)] = knn_update([neighbor_dists(j,:), dists(j,1:numel(inds))], [neighbor_inds(j,:), inds], k);
        
    end
    
    % estimate the accuracy in each iteration
    total = 0;
    for j = 1:num_queries
        total = total + length(intersect(actual_nn(j,:), neighbor_inds(j,:)));
    end
    
    dist_acc = sum(abs(neighbor_dists(:))) / sum(abs(actual_dists(:)));

    rank_acc = total/(num_queries*k);

    fprintf('Iteration %d: Rank accuracy %g, Dist accuracy: %g, with %d hash evals and %g total evals.\n', i, rank_acc, dist_acc, ...
        hash_evals, (hash_evals + search_evals)/(num_references*num_queries));
    
end

dist_evals = (p*p + num_references*p + num_queries*p + search_evals)/(num_queries*num_references);

end

