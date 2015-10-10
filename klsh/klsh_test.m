

clear; clc; close all;
addpath '../src'
addpath '../src/kernels/';

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

testfilename = '~/susy.icml.tst.X.bin';
m = 499999;

filename = '~/susy.icml.trn.X.bin';
n = 4499999;
dim = 18;


train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

test = binread_array(testfilename, m*dim);
test = reshape(test, dim, m);

data = datasample(train, N, 2, 'Replace', false);
queries = datasample(test, Nq, 2, 'Replace', false);

rng('shuffle')

% d = 2;
% N = 10000;
% Nq = 100;
% 
% data = [randn(d,N/2), randn(d,N/2) + repmat([10;10], 1, N/2)];
% queries = randn(d, Nq);

k = 10;



%% kernel function

% h = 0.15;
% h = 0.22;
% h = 0.1;
% kernel = @(x, y) rbf(x, y, h);
% kernel_dist = @(x,y) rbf_dist(x, y, h);

h = 1;
c = 0;
p = 2;
kernel = @(x, y) poly(x, y, h, c, p);

% kernel_dist = @(x, y) poly_dist(x, y, h, c, p);

% kernel = @(x, y) cosine_kernel(x, y);
% kernel_dist = @(x, y) cosine_kernel_dist(x,y);

% h = 10;
% c = 0;
% kernel = @(x,y) hyptan_kernel(x,y,h,c);

% search parameters

%%
% exact eval

fprintf('Doing exact computation.\n');
[actual_dists, actual_nn] = knn_update(kernel(queries, data), repmat(1:N, size(queries,2),1), k);
fprintf('Exact computation done.\n');



%% 

% parameters
p = 300;
b = 30;
B = 2;
t = 30;

num_runs = 3;

epsilon = 1;
M = floor(N^(1/(1+epsilon)));

rank_acc = zeros(num_runs, 1);
dist_acc = zeros(num_runs, 1);
search_evals = zeros(num_runs, 1);

for i = 1:num_runs

    [rank_acc(i), dist_acc(i), search_evals(i)] = klsh(queries, data, k, b, B, M, p, t, kernel, actual_dists, actual_nn);

end

fprintf('\n\n===================================\n\n');
fprintf('Results: (min, avg, max) over %d runs\n\n', num_runs);

fprintf('Rank accuracy: %g, %g, %g\n', min(rank_acc), mean(rank_acc), max(rank_acc));
fprintf('Distance accuracy: %g, %g, %g\n', min(dist_acc), mean(dist_acc), max(dist_acc));
fprintf('Search Evaluations: %g, %g, %g\n', min(search_evals), mean(search_evals), max(search_evals));







