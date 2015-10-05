

clear; clc; close all;

addpath('../src/');

%% generate data

filename = '~/covtype.libsvm.trn.X.bin';
n = 499999;
dim = 54;
train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

% read test
filename = '~/covtype.libsvm.tst.X.bin';
m = 80012;
test = binread_array(filename, m*dim);
test = reshape(test, dim, m);

% sample data
N = 499999;
Nq = 100;
data = datasample(train, N, 2, 'Replace', false);
queries = datasample(test, Nq, 2, 'Replace', false);


h = 0.22;

% search parameters

max_iter = 100;
tolerance = 0.99;
max_dists = 1;
max_points_per_node = 100;
k = 10;
maxLevel = 30;
num_backtracks = 2;

%% compute exact nearest neighbors 

exact_nn = kknn(data, 1:N, queries, h, k, N);


%%

test_nn = ones(Nq, k);
points = zeros(Nq, k);

iter = 0;
acc = 0;

total_dists = 0;
dist_frac = 0;

fprintf('Partial backtracking: \n\n');

while (iter <= max_iter && acc < tolerance && dist_frac < max_dists)

    % build the new tree
    tree = bsttree_vp(data, 1:N, max_points_per_node, maxLevel, h, 0, 0);
    
    [new_nn,deval] = PartialBacktracking(tree, queries, h, 1:Nq, data, k, points, test_nn, 0, num_backtracks);
    test_nn= new_nn;
    points = test_nn;
    total_dists = total_dists + deval;

    % now, estimate accuracy
    
    suml = 0;
    % calculate accuracy by comparing neighbors found
    for i = 1:Nq
        suml = suml + length(intersect(points(i,:), exact_nn(i,:)));
    end
    
    % print accuracy
    acc = suml/(Nq*k);
    fprintf('Accuracy: %f\n', suml/(Nq*k));
    fprintf('Num dist evals: %g\n\n', total_dists/(N * Nq));   
    
    iter = iter + 1;
    
    dist_frac = total_dists / (N * Nq);
    
end % loop over random trees


%% Compare against existing code

fprintf('\n\n========================================================\n\n');
fprintf('No backtracking:\n\n');

iter = 0;
total_dists = 0;
acc = 0;
test_nn = ones(Nq, k);
points = zeros(Nq, k);
dist_frac = 0;


while (iter <= max_iter && acc < tolerance && dist_frac < max_dists)
    

    % build the new tree
    tree = bsttree_vp(data, 1:N, max_points_per_node, maxLevel, h, 0, 0);
    
    [new_nn,deval] = travtree2n(tree, queries, h, 1:Nq, data, k, points, test_nn, 0);
    test_nn= new_nn;
    points = test_nn;
    total_dists = total_dists + deval;

    % now, estimate accuracy
    
    suml = 0;
    % calculate accuracy by comparing neighbors found
    for i = 1:Nq
        suml = suml + length(intersect(points(i,:), exact_nn(i,:)));
    end
    
    % print accuracy
    acc = suml/(Nq*k);
    fprintf('Accuracy: %f\n', suml/(Nq*k));
    fprintf('Num dist evals: %g\n\n', total_dists/(N * Nq));   
    
    iter = iter + 1;

    dist_frac = total_dists / (N * Nq);

    
end % loop over random trees









