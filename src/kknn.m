
function points = kknn(data, indi, query, sigma, k, n)
%--------------------------------------------------------------------------
% KKNN computes k-nearest neighbors for kernels
%   Input 
%       data - data values
%       indi - global ids of the data points
%       query - point whose neighbors are to be computed
%       sigma - bandwidth
%       k - number of nearest neighbors that have to be computed
%       n - total number of points
%
%   Output 
%       points - global ids of nearest neighbors
%--------------------------------------------------------------------------
% in the event that number of neighbors is greater than n
if(k > n)
    points = horzcat(data, data(:,1:(k - n)));
    return;
end

% linear search through all the points
dist = distk(data(:, indi), query, sigma);

len = size(query, 2);

% if there is 1 query point or many
if(len == 1)
    % sort distance and report k NN
    [~,ind] = sort(dist);
    points = indi(ind(1:k));
else
    % sort distance and report k NN
    [~,ind] = sort(dist);
    ind = ind';
    points = indi(ind(:,1:k));
end

end
