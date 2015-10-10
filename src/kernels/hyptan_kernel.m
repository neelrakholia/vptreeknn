function [ kval ] = hyptan_kernel( x, y, h, c )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

kval = tanh(x'*y/h + c * ones(size(x,2),size(y,2)));

end

