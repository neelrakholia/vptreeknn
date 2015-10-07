
clear; clc; close all;
addpath './kernels/';

%% load data

rng(1)

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
N = 200000;
Nq = 1000;
data = datasample(train, N, 2, 'Replace', false);
queries = datasample(test, Nq, 2, 'Replace', false);

rng('shuffle')

%% kernel function

h = 0.22;
% kernel = @(x, y) rbf(x, y, h);
kernel = @(x, y) poly(x, y, 1, 0, 10);
% kernel = @(x,y) rbf_dist(x, y, h);
% kernel = @(x, y) poly_dist(x, y, 1, 0, 10);

% search parameters

max_iter = 100;
tolerance = 0.99;
max_dists = 1;
max_points_per_node = 1000;
k = 10;
maxLevel = 12;

%% compute exact nearest neighbors 

exact_nn = kknn(data, 1:N, queries, kernel, k, N);

%% do tree search

[ rank_acc, dist_acc, tree_evals, search_evals, neighbors ] = randomvp_search( data, queries, kernel, ...
    k, exact_nn, max_iter, tolerance, max_dists, max_points_per_node, maxLevel );




