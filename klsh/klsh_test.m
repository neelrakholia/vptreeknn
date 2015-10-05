
addpath '~/vptreeknn/src'
clear

% for debugging
rng(1);

% read train
filename = '~/covtype.libsvm.trn.X.bin';
num_references = 499999;
dim = 54;
train = binread_array(filename, num_references*dim);
train = reshape(train, dim, num_references);

% read test
filename = '~/covtype.libsvm.tst.X.bin';
num_queries = 80012;
test = binread_array(filename, num_queries*dim);
test = reshape(test, dim, num_queries);

% sample data
num_references = 10000;
num_queries = 100;
train = datasample(train, num_references, 2, 'Replace', false);
test = datasample(test, num_queries, 2, 'Replace', false);

% sample data to form kernel matrix
p = 300;
b = 30;
B = 15;
k = 10;

epsilon = 1;
M = floor(num_references^(1/(1+epsilon)));

h = 0.22;
kernelfun = @(q,r) distk(q, r, h);

[acc, dists] = klsh(test, train, k, b, B, M, p, kernelfun);





