% constructs a tree out documents based on proximity

rng(100);

% add appropriate paths 
clear globals; clc; clear all;
addpath('src/')
addpath('..')

% read data from crude and livestock
crude = readfiles('data/Reuters21578-Apte-90Cat/training/crude');
livestock = readfiles('data/Reuters21578-Apte-90Cat/training/livestock');
% combine data
data = cat(2,crude,livestock);

% get labels for data
labels = ones(length(data),1);
labels(1:length(crude)) = 0;

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

% get global indices for data
global_id = 1:20;%length(train);

% brute force search
tic
nn = kknn(train,global_id,test(371),0.5,10,length(train));
toc
