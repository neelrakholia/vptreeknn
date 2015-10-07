function [ rank_acc, dist_acc, tree_frac, search_frac, points ] = randomvp_search( data, queries, kernel, k, exact_nn, max_iter, tolerance, max_dists, ...
    max_points_per_node, maxLevel)
%randomvp_search Wrapper function for KNN search for 
%
% INPUTS:
% - data -- the reference set
% - queries -- the query set
% - kernel -- handle for the similarity function
% - k -- the number of neighbors to find
% - exact_nn -- the exact neighbors for accuracy computations
% - max_iter -- number of iterations to use 
% - tolerance -- rank accuracy tolerance to terminate iterations
% - max_dists -- maximum fraction of distance evaluations to perform
% - max_points_per_node -- leaf size
% - max level -- maximum tree depth
%
% OUTPUTS:
% - rank_acc -- the fraction of true NN we recovered
% - dist_acc -- the ratio of the similarities found to the true Nq * k 
%   most similar points
% - tree_frac -- fraction of total distances in tree construction
% - search_frac -- fraction of total distances in search
% - points -- Nq x k array of NN indices



N = size(data,2);
Nq = size(queries, 2);

test_nn = ones(Nq, k);
points = zeros(Nq, k);

iter = 0;
rank_acc = 0;

total_dists = 0;
tree_dists = 0;
total_search_evals = 0;
total_search_frac = 0;

true_dists = zeros(Nq, k);
for i = 1:Nq
    true_dists(i,:) = kernel(queries(:,i), data(:, exact_nn(i,:)));
end


while (iter <= max_iter && rank_acc < tolerance && total_search_frac < max_dists)
    
    % build the new tree
    tree = bsttree_vp(data, 1:N, max_points_per_node, maxLevel, kernel, 0, tree_dists);
    % we know that VP tree construction requires N distance evaluations
    this_tree_dists = N; 
    tree_dists = tree_dists + this_tree_dists;

    % update nearest neighbors with tree search
    [new_nn, search_dists] = travtree2n(tree, queries, 1:Nq, data, k, points, test_nn, 0);
    test_nn = new_nn;
    points = test_nn;
    total_dists = total_dists + search_dists + this_tree_dists;

    % now, estimate accuracy
    
    suml = 0;
    these_dists = zeros(Nq, k);

    % calculate accuracy by comparing neighbors found
    % TODO: need to not recompute these moving forward
    for i = 1:Nq
        suml = suml + length(intersect(points(i,:), exact_nn(i,:)));
        these_dists(i,:) = kernel(queries(:,i), data(:, points(i,:)));
    end


    dist_frac = total_dists / (N * Nq);
    tree_frac = this_tree_dists / (N * Nq);
    search_frac = search_dists / (N * Nq);
    total_search_evals = total_search_evals + search_dists;
    total_search_frac = total_search_evals / (N * Nq);

    % print accuracy
    
    rank_acc = suml/(Nq*k);
    dist_acc_frac = sum(these_dists(:)) / sum(true_dists(:));
    % if we're doing -1*distance, then we need to flip this over for it to  
    % be meaningful
    dist_acc = min(dist_acc_frac, 1/dist_acc_frac);

    fprintf('Iteration %d. Rank acc: %g, Dist acc: %g, Tree evals: %g, Search evals: %g, Total evals: %g\n\n', ...
        iter, rank_acc, dist_acc, tree_frac, search_frac, dist_frac);

    iter = iter + 1;
    
end % loop over random trees




end

