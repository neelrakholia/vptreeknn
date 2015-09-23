% This script searches for NN on susy dataset

% add appropriate paths 
clear globals; clc; clear all;
addpath('src/')
addpath('..')

% read train
filename = 'data/susy.icml.trn.X.bin';
n = 4499999;
dim = 18;
train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

% read test
filename = 'data/susy.icml.tst.X.bin';
m = 499999;
test = binread_array(filename, m*dim);
test = reshape(test, dim, m);

% sample data
n = 1000000;
m = 200000;
train = datasample(train, n, 2, 'Replace', false);
test = datasample(test, m, 2, 'Replace', false);

randomvp(test, train, m, n, 100, 100, 0.15, 30000, 12, 10000);