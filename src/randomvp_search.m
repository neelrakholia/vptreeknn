function [ rank_acc, dist_acc_frac, total_search_frac, iter,  vp_nn_inds ] = randomvp_search( data, queries, kernel, kernel_dist, k, exact_nn, max_iter, tolerance, max_dists, ...
    max_points_per_node, maxLevel, num_backtracks)
%randomvp_search Wrapper function for KNN search for 
%
% INPUTS:
% - data -- the reference set
% - queries -- the query set
% - kernel -- handle for the similarity function
% - kernel_dist -- handle for distance version (K(x,x) + K(y,y) - 2 K(x,y))
% - k -- the number of neighbors to find
% - exact_nn -- the exact neighbors for accuracy computations
% - max_iter -- number of iterations to use 
% - tolerance -- rank accuracy tolerance to terminate iterations
% - max_dists -- maximum fraction of distance evaluations to perform
% - max_points_per_node -- leaf size
% - max level -- maximum tree depth
% - num_backtracks -- number of backtracks to do in search
%
% OUTPUTS:
% - rank_acc -- the fraction of true NN we recovered
% - dist_acc -- the ratio of the similarities found to the true Nq * k 
%   most similar points
% - search_frac -- fraction of total distances in search
% - num_iterations -- number of search iterations
% - vp_nn_inds -- Nq x k array of NN indices

global do_plot


N = size(data,2);
Nq = size(queries, 2);

vp_nn_inds = zeros(Nq,k);
vp_nn_dists = -inf*ones(Nq,k);

iter = 0;
rank_acc = 0;

total_search_evals = 0;
total_search_frac = 0;

true_dists = zeros(Nq, k);
for i = 1:Nq
    true_dists(i,:) = kernel(queries(:,i), data(:, exact_nn(i,:)));
end

dist_acc_frac = 0;

while (iter <= max_iter && rank_acc < tolerance && dist_acc_frac < tolerance && total_search_frac < max_dists)


if (do_plot)
    figure()
    scatter(queries(1,1), queries(2,1), 200, 'm', 'x');
    hold on;                
    scatter(data(1,exact_nn(1)), data(2, exact_nn(2)), 200, 'm', 'd');
end

    % build the new tree
    tree = bsttree_vp(data, 1:N, max_points_per_node, maxLevel, kernel, kernel_dist, 0, 0);
    % we know that VP tree construction requires N log(N/leaf_size) distance evaluations

    % update nearest neighbors with tree search
    if (num_backtracks > 0)
        [new_nn_inds, new_nn_dists, num_search_dists] = PartialBacktracking(tree, queries, data, k, vp_nn_inds, vp_nn_dists, num_backtracks);
    else
        [new_nn_inds, new_nn_dists, num_search_dists] = travtree2n(tree, queries, data, k, vp_nn_inds, vp_nn_dists, 0);
    end    

    vp_nn_inds = new_nn_inds;
    vp_nn_dists = new_nn_dists;
    
    % now, estimate accuracy
    
    suml = 0;
    % calculate accuracy by comparing neighbors found
    % TODO: need to not recompute these moving forward
    for i = 1:Nq
        suml = suml + length(intersect(vp_nn_inds(i,:), exact_nn(i,:)));
    end

    search_frac = num_search_dists / (N * Nq);
    total_search_evals = total_search_evals + num_search_dists;
    total_search_frac = total_search_evals / (N * Nq);

    % print accuracy
    
    rank_acc = suml/(Nq*k);
    dist_acc_frac = sum(abs(vp_nn_dists(:))) / sum(abs(true_dists(:)));
    
    fprintf('Iteration %d. Rank acc: %g, Dist acc: %g, Evals in this iter: %g, Total evals: %g\n\n', ...
        iter, rank_acc, dist_acc_frac, search_frac, total_search_frac);

    iter = iter + 1;
    
end % loop over random trees

end

