
<<<<<<< HEAD
addpath '/src'
=======
addpath '~/vptreeknn/src'
clear
>>>>>>> origin/master

% for debugging
rng(1);

% read train
filename = 'data/susy.icml.trn.X.bin';
num_references = 4499999;
dim = 18;
train = binread_array(filename, num_references*dim);
train = reshape(train, dim, num_references);

% read test
filename = 'data/susy.icml.tst.X.bin';
num_queries = 499999;
test = binread_array(filename, num_queries*dim);
test = reshape(test, dim, num_queries);

% sample data
num_references = 200000;
num_queries = 1000;
train = datasample(train, num_references, 2, 'Replace', false);
test = datasample(test, num_queries, 2, 'Replace', false);

% sample data to form kernel matrix
<<<<<<< HEAD
p = 200;
b = 60;
B = 3000;
k = 100;
=======
p = 300;
b = 30;
B = 15;
k = 10;
>>>>>>> origin/master

epsilon = 1;
M = floor(num_references^(1/(1+epsilon)));

h = 0.15;
kernelfun = @(q,r) distk(q, r, h);

[acc, dists] = klsh(test, train, k, b, B, M, p, kernelfun);





