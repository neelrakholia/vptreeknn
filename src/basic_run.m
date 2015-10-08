
clear; clc; close all;
addpath './kernels/';

%% load data

rng(1)

filename = '~/covtype.libsvm.trn.X.bin';
n = 499999;
dim = 54;

testfilename = '~/covtype.libsvm.tst.X.bin';
m = 80012;

% filename = '~/susy.icml.tst.X.bin';
% m = 499999;
% 
% filename = '~/susy.icml.trn.X.bin';
% n = 4499999;
% dim = 18;



train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

test = binread_array(testfilename, m*dim);
test = reshape(test, dim, m);

N = 200000;
Nq = 1000;

data = datasample(train, N, 2, 'Replace', false);
queries = datasample(test, Nq, 2, 'Replace', false);

rng('shuffle')

% d = 2;
% N = 10000;
% Nq = 100;
% 
% data = [randn(d,N/2), randn(d,N/2) + repmat([10;10], 1, N/2)];
% queries = randn(d, Nq);


%% kernel function

h = 0.22;
kernel = @(x, y) rbf(x, y, h);
kernel_dist = @(x,y) rbf_dist(x, y, h);

% kernel = @(x, y) poly(x, y, 1, 0, 10);
% kernel_dist = @(x, y) poly_dist(x, y, 1, 0, 10);

% search parameters

max_iter = 100;
tolerance = 0.99;
max_dists = 1;
max_points_per_node = 1000;
k = 10;
maxLevel = 12;
num_backtracks = 0;

%% compute exact nearest neighbors 

fprintf('Doing exact computation.\n');
% exact_nn = kknn(data, 1:N, queries, kernel, k, N);
[~, exact_nn]= knn_update(kernel(queries, data), repmat(1:N, size(queries,2),1), k);
fprintf('Exact computation done.\n');

%% do tree search

global do_plot
do_plot = false;

[ rank_acc, dist_acc, search_evals, neighbors ] = randomvp_search( data, queries, kernel, kernel_dist, ...
    k, exact_nn, max_iter, tolerance, max_dists, max_points_per_node, maxLevel, num_backtracks );




