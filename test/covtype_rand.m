% This script searches for NN on covtype dataset

% add appropriate paths 
clear globals; clc; clear all;
addpath('src/')
addpath('..')

% read train
filename = 'data/covtype.libsvm.trn.X.bin';
n = 499999;
dim = 54;
train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

% read test
filename = 'data/covtype.libsvm.tst.X.bin';
m = 80012;
test = binread_array(filename, m*dim);
test = reshape(test, dim, m);

% sample data
n = 200000;
m = 1000;
train = datasample(train, n, 2, 'Replace', false);
test = datasample(test, m, 2, 'Replace', false);

randomvp(test, train, m, n, 10, 100, 0.22, 1000, 12, 10);