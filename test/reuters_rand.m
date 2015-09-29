% constructs a tree out documents based on proximity

rng(100);

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

fprintf('finished reading data\n');

% combine data
data = cat(2,crude,livestock);
data = cat(2,data,earn);
data = cat(2,data,acq);

% get labels for data
% crude = 0;
% livestock = 1;
% earn = 2;
labels = ones(length(data),1);
labels(1:length(crude)) = 0;
labels(length(crude) + length(livestock) + 1:...
    length(crude) + length(livestock) + length(earn)) = 2;
labels(length(crude) + length(livestock) + length(earn) + 1:...
    end) = 3;

% percentage split for test and train
per = 0.2;

% test/train split. per % data is train and  1 - per % is test
ind = randperm(length(data));
train_ind = ind(1:floor(length(data)*per));
test_ind = ind(floor(length(data)*per)+1:end);
train = data(train_ind);
train_lab = labels(train_ind);
test = data(test_ind);
test_lab = labels(test_ind);

%%

% compile mex file
mex -outdir ../src/ COPTIMFLAGS='-O3 -DNDEBUG' ../src/ssk_dyn_mex.c 
mex -outdir ../src/ COPTIMFLAGS='-O3 -DNDEBUG' ../src/wsk_mex.c 
% mex -outdir ../src/ COPTIMFLAGS='-pg' ../src/ssk_dyn_mex.c 
% mex -outdir ../src/ COPTIMFLAGS='-pg' ../src/wsk_mex.c 
fprintf('finished compiling\n');

%%

num_train = floor(numel(train)/2);
num_test = floor(numel(test)/2);
lambda = 0.5;
order = 3;

% run random NN algo and see how it compares with quad search
randomvp(train(1:num_train), test(1:num_test), num_train, num_test, 10, 5, lambda, ...
     50, 8, 1, train_lab(1:num_train), test_lab(1:num_test))

% tic
% K = wsk_mex(train(1:num_train), test(1:num_test), lambda, order);
% toc
% 
% tic 
% K_exact = wsk(train(1:num_train), test(1:num_test), lambda, order);
% toc
% 
% fprintf('Error: %g\n', norm(K - K_exact));





