% constructs a tree out documents based on proximity

rng(1);

% add appropriate paths 
clear globals; clc; clear all;
addpath('../src/')
addpath('../data/')
% addpath('..')

% read data from crude and livestock
crude = readfiles('../data/Reuters21578-Apte-90Cat/training/crude');
livestock = readfiles('../data/Reuters21578-Apte-90Cat/training/livestock');
earn = readfiles('../data/Reuters21578-Apte-90Cat/training/earn');
acq = readfiles('../data/Reuters21578-Apte-90Cat/training/acq');
unknown = readfiles('../data/Reuters21578-Apte-90Cat/training/unknown');
moneyfx = readfiles('../data/Reuters21578-Apte-90Cat/training/money-fx');
interest = readfiles('../data/Reuters21578-Apte-90Cat/training/interest');
grain = readfiles('../data/Reuters21578-Apte-90Cat/training/grain');
trade = readfiles('../data/Reuters21578-Apte-90Cat/training/trade');
corn = readfiles('../data/Reuters21578-Apte-90Cat/training/corn');
oilseed = readfiles('../data/Reuters21578-Apte-90Cat/training/oilseed');
ship = readfiles('../data/Reuters21578-Apte-90Cat/training/ship');
moneysupply = readfiles('../data/Reuters21578-Apte-90Cat/training/money-supply');
dlr = readfiles('../data/Reuters21578-Apte-90Cat/training/dlr');
sugar = readfiles('../data/Reuters21578-Apte-90Cat/training/sugar');
natgas = readfiles('../data/Reuters21578-Apte-90Cat/training/nat-gas');
wheat = readfiles('../data/Reuters21578-Apte-90Cat/training/wheat');
vegoil = readfiles('../data/Reuters21578-Apte-90Cat/training/veg-oil');
gnp = readfiles('../data/Reuters21578-Apte-90Cat/training/gnp');
soybean = readfiles('../data/Reuters21578-Apte-90Cat/training/soybean');
gold = readfiles('../data/Reuters21578-Apte-90Cat/training/gold');
coffee = readfiles('../data/Reuters21578-Apte-90Cat/training/coffee');

fprintf('finished reading data\n');

% combine data
data = cat(2,crude,livestock);
data = cat(2,data,earn);
data = cat(2,data,acq);
data = cat(2,data,unknown);
data = cat(2,data,moneyfx);
data = cat(2,data,interest);
data = cat(2,data,grain);
data = cat(2,data,trade);
data = cat(2,data,corn);
data = cat(2,data,oilseed);
data = cat(2,data,ship);
data = cat(2,data,moneysupply);
data = cat(2,data,dlr);
data = cat(2,data,sugar);
data = cat(2,data,natgas);
data = cat(2,data,wheat);
data = cat(2,data,vegoil);
data = cat(2,data,gnp);
data = cat(2,data,soybean);
data = cat(2,data,gold);
data = cat(2,data,coffee);

% percentage split for test and train
per = 101/length(data);

% test/train split. per % data is train and  1 - per % is test
ind = randperm(length(data));
train_ind = ind(1:floor(length(data)*per));
test_ind = ind(floor(length(data)*per)+1:end);
train = data(train_ind);
test = data(test_ind);

%%

% compile mex file
mex -outdir ../src/ COPTIMFLAGS='-O3 -DNDEBUG' ../src/ssk_dyn_mex.c 
mex -outdir ../src/ COPTIMFLAGS='-O3 -DNDEBUG' ../src/wsk_mex.c 
mex -outdir ../src/ COPTIMFLAGS='-O3 -DNDEBUG' ../src/wsk_dist_mex.c 
% mex -outdir ../src/ COPTIMFLAGS='-pg' ../src/ssk_dyn_mex.c 
% mex -outdir ../src/ COPTIMFLAGS='-pg' ../src/wsk_mex.c 
fprintf('finished compiling\n');

%%

N = 2000;
Nq = 100;

data = test(1:N);
queries = train(1:Nq);

lambda = 0.5;
order = 2;
kernel = @(x, y) wsk_mex(x, y, lambda, order);
kernel_dist = @(x, y) wsk_dist_mex(x, y, lambda, order);

% kernel = @(x, y) cosine_kernel(x, y);
% kernel_dist = @(x, y) cosine_kernel_dist(x,y);

% h = 1;
% c = 0;
% kernel = @(x,y) hyptan_kernel(x,y,h,c);
% kernel_dist = @(x,y) hyptan_kernel_dist(x,y,h,c);

% search parameters

max_iter = 20;
tolerance = 0.90;
max_dists = 1;
max_points_per_node = 50;
k = 10;
maxLevel = 12;
% Important: this needs to be smaller than the depth of the tree
num_backtracks = 0;

num_runs = 1;

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