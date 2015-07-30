% This script searches for NN on gaussian set of points

% add appropriate paths 
clear globals;clc; clear all;
addpath('src/')

% generate points, n = database points, m = query points
n = 2^14;
m = 400;
dim = 8;

% number of nearest neighbors, and number of trees to generate
K = 10;
ntree = 20;

% random generation of database and query points
point_distribution = 'gaussian';
r = generate_points(dim, n, point_distribution);
q = generate_points(dim, m, point_distribution);

randomvp(q, r, m, n, 10, 100, 2, 2^7, 12, 100);