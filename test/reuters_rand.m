% constructs a tree out documents based on proximity

rng(100);

% compile mex file
mex -outdir src/ src/ssk_dyn_mex.c 
mex -outdir src/ src/wsk_mex.c 

% add appropriate paths 
clear globals; clc; clear all;
addpath('src/')
addpath('..')

% read data from crude and livestock
crude = readfiles('data/Reuters21578-Apte-90Cat/training/crude');
livestock = readfiles('data/Reuters21578-Apte-90Cat/training/livestock');
earn = readfiles('data/Reuters21578-Apte-90Cat/training/earn');
acq = readfiles('data/Reuters21578-Apte-90Cat/training/acq');
% combine data
data = cat(2,crude,livestock);
data = cat(2,data,earn);
data = cat(2,data,acq);

% get labels for data
% crude = 0
% livestock = 1
% earn = 2
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

% run random NN algo and see how it compares with quad search
randomvp(train(1:100), test(1:400), 100, 400, 10, 3, 0.5, ...
    40, 5, 1, train_lab(1:100), test_lab(1:400))




