function [datal, datar, indl, indr, rad, cent, diseval] = classify_vp(data, ...
    indi, nsize, kernel, diseval)
%--------------------------------------------------------------------------
% CLASSIFY_VP Computes the left data, the right data, and the radius of
% left node
%   Input 
%       data - database points to be organized into a vp-tree
%       indi - global ids of database points
%       nsize - number of points
%       kernel - handle for kernel function
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
% dist = distk(data, bestp, kernel);
dist = kernel(data, bestp);

% compute midpoint
mid = ceil(nsize/2);

% sort distance and find the elements in each node
[~,ind] = sort(dist);
datal = data(:,ind(1:mid));
indl = indi(ind(1:mid));
datar = data(:,ind(mid+1:end));
indr = indi(ind(mid+1:end));
cent = bestp;

% figure();
% scatter(datal(1,:), datal(2,:), 20, 'bx')
% hold on;
% scatter(datar(1,:), datar(2,:), 20, 'ro')
% scatter(bestp(1), bestp(2), 40, 'kd')

% compute radius
% rad = distk(bestp, datal(:, end), kernel);
% rad = kernel(bestp, datal(:,end));
% using the median instead -- helps with big splits?
rad = median(dist);

end