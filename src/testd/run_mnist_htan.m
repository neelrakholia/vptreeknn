
clear; clc; close all;
addpath ..
addpath 'kernels/';

%% load data

rng(1)

% filename = '~/covtype.libsvm.trn.X.bin';
% n = 499999;
% dim = 54;
% 
% testfilename = '~/covtype.libsvm.tst.X.bin';
% m = 80012;

% filename = '~/uniform_64d_6id_500k.bin';
% n = 499999;
% dim = 64;
% 
% data_in = binread_array(filename, n*dim);
% data_in = reshape(data_in, dim, n);

N = 200000;
Nq = 1000;


% data = data_in(:,1:N);
% queries = data_in(:,(end-Nq+1):end);

testfilename = 'mnist_test.csv';
m = 10000;

filename = 'mnist_train.csv';
n = 60000;
dim = 784;

% read file and remove label
train = csvread(filename);
train = train(:,2:end)';

test = csvread(testfilename);
test = test(:,2:end)';

data = train./255;
queries = datasample(test, Nq, 2, 'Replace', false)./255;

rng('shuffle')

% d = 2;
% N = 10000;
% Nq = 100;
% 
% data = [randn(d,N/2), randn(d,N/2) + repmat([10;10], 1, N/2)];
% queries = randn(d, Nq);


%% kernel function

% h = 0.15;
% h = 0.22;
% h = 0.1;
% kernel = @(x, y) rbf(x, y, h);
% kernel_dist = @(x,y) rbf_dist(x, y, h);

h = 10;
c = 0;
kernel = @(x, y) hyptan_kernel(x, y, h, c);
kernel_dist = @(x, y) hyptan_kernel_dist(x, y, h, c);

% kernel = @(x, y) cosine_kernel(x, y);
% kernel_dist = @(x, y) cosine_kernel_dist(x,y);

% h = 1;
% c = 0;
% kernel = @(x,y) hyptan_kernel(x,y,h,c);
% kernel_dist = @(x,y) hyptan_kernel_dist(x,y,h,c);

% search parameters

max_iter = 100;
tolerance = 0.99;
max_dists = 1;
max_points_per_node = 1000;
k = 100;
maxLevel = 12;
% Important: this needs to be smaller than the depth of the tree
num_backtracks = 0;

num_runs = 5;

%% compute exact nearest neighbors 

fprintf('Doing exact computation.\n');
% exact_nn = kknn(data, 1:N, queries, kernel, k, N);
[~, exact_nn]= knn_update(kernel(queries, data), repmat(1:N, size(queries,2),1), k);
fprintf('Exact computation done.\n');

%% do tree search

global do_plot
do_plot = false;

rank_acc = zeros(num_runs, 1);
dist_acc = zeros(num_runs, 1);
search_evals = zeros(num_runs, 1);
num_iters = zeros(num_runs, 1);

for i = 1:num_runs

[ rank_acc(i), dist_acc(i), search_evals(i), num_iters(i), neighbors ] = randomvp_search( data, queries, kernel, kernel_dist, ...
    k, exact_nn, max_iter, tolerance, max_dists, max_points_per_node, maxLevel, num_backtracks );

end

tree_evals = num_iters * N * ceil(log2(N / max_points_per_node)) + 2 * N;

fprintf('\n\n===================================\n\n');
fprintf('Results: (min, avg, max) over %d runs\n\n', num_runs);

fprintf('Rank accuracy: %g, %g, %g\n', min(rank_acc), mean(rank_acc), max(rank_acc));
fprintf('Distance accuracy: %g, %g, %g\n', min(dist_acc), mean(dist_acc), max(dist_acc));
fprintf('Num iterations: %d, %g, %d\n', min(num_iters), mean(num_iters), max(num_iters));
fprintf('Search Evaluations: %g, %g, %g\n', min(search_evals), mean(search_evals), max(search_evals));
fprintf('Tree Evaluations: %d, %g, %d\n', min(tree_evals), mean(tree_evals), max(tree_evals));



