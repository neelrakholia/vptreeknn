function [datal, datar, indl, indr, rad, cent, diseval] = classify_vp(data, ...
    indi, nsize, sigma, diseval)
%--------------------------------------------------------------------------
% CLASSIFY_VP Computes the left data, the right data, and the radius of
% left node
%   Input 
%       data - database points to be organized into a vp-tree
%       indi - global ids of database points
%       nsize - number of points
%       sigma - kernel bandwidth parameter
%       diseval - variable to keep track of distance evaluations
%
%   Output 
%       datal - data in the left node
%       datar - data in the right node
%       indl - global ids of datal
%       indr - global ids of datar
%       rad - radius of left node
%       cent - selected vantage point
%       diseval - updated distance evaluations
%--------------------------------------------------------------------------
% the number of random points to select
rand = 1; 

% update distance evaluations
diseval = diseval + rand*nsize;

% select random points
perm = randperm(nsize);
index = perm(1:rand);

% linear search through all the points
cent = data(:, index(rand));
bestp = cent;
dist = distk(data, bestp, sigma);

% compute midpoint
mid = ceil(nsize/2);

% sort distance and find the elements in each node
[~,ind] = sort(dist);
datal = data(:,ind(1:mid));
indl = indi(ind(1:mid));
datar = data(:,ind(mid+1:end));
indr = indi(ind(mid+1:end));
cent = bestp;

% compute radius
rad = distk(bestp, datal(:, end), sigma);
    
end