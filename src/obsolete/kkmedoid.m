% NOTE: THIS FILE IS NO LONGER USED IN OUR METHOD. FOR INFORMATIONAL
% PURPOSES ONLY

% Taken from: https://github.com/SCRFpublic/DGSA/blob/master/kmedoids.m
% This function applies a (kernel) k-medoid algorithm (PMA) 
% in the Feature Space defined by the kernel matrix K. 
 
% Author: Celine Scheidt
% Date: April 2009
% Updated: February 2013


function Clustering = kkmedoid(Xd,nbclusters,nbrep,K)

%% Input Parameters
%   - Xd: location of the points to be plotted (from MDS). Rows of Xd are 
%         the coordinates in p-dimensional space. 
%         Dimension: Number of models (n) x p
%   - nbclusters: number of clusters to construct
%   - nbrep: (optional). Number of clustering performed. The best cluster 
%                       configuration is returned
%   - K: Kernel matrix (if not provided, k-medoid is performed instead of
%        kernel k-medoid


%% Output Parameters 
%   - Clustering: Results of the clustering, which contains:
%        - label: is a vector of length n, which contains the cluster index that each model belongs to
%        - medoids: is a vector of length nbcluster containing the index
%                   of the medoids
%        - weights: is a vector of ength nbcluster containing the number
%                   of models in each cluster.
%

maxIterations = 50;
npoints = size(Xd,1);
minDistBest = Inf;

if nargin < 3
    nbrep = 1;
end

% Definition of the distance pairwise distances between models
if nargin < 4
    D = squareform(pdist(Xd)); % If kernel matrix not provided, Euclidean distance is used
else 
    D = dist_feature(K,1:npoints,1:npoints); % distance in Feature space
end

global labelBest;
global currentMedoidsBest;

for iter = 1:nbrep % nbrep clustering are performed, the best is returned
    
    % 1. Initalize: randonly select nbclusters of the npoints data points as the medoids
    initMedoids = randsample(npoints,nbclusters);
    
    % 2. Associate each data point to the closest medoid
    [minDistInit, label] = min(D(initMedoids,:));

    currentMedoids = initMedoids;
    minDistCurrent = minDistInit;
    
    label_prev = NaN(1,npoints);
    nbIter = 0;
    
    while any(label ~= label_prev) && nbIter < maxIterations % while cluster configuration is changing and maxIteration not reached
              
        label_prev = label;
        
        % 3. For each medoid m
        for m = 1:nbclusters
            
            NoMedoid = setdiff(1:npoints,currentMedoids);
            NewMedoids = currentMedoids;
            
             % For each non-medoid data point o
            for o = 1:length(NoMedoid)               
                % Swap m and o and compute the cost of the configuration
                NewMedoids(m) = NoMedoid(o);
                [minDist, label] = min(D(NewMedoids,:));
                cost = sum(minDist) - sum(minDistCurrent);
                
                if cost < 0  % 4. Select the configuration with the lowest cost
                    currentMedoids(m) = NoMedoid(o);
                    [minDistCurrent, label] = min(D(currentMedoids,:));
                end
            end          
        end
        
        nbIter = nbIter+1;
    end
    
    currentMedoids = sort(currentMedoids);
    [minDist, label] = min(D(currentMedoids,:));
        labelBest = label;
        currentMedoidsBest = currentMedoids;
    
    % Return the best clustering configuration among the nbrep tested
     if sum(minDist) < sum(minDistBest)  
        minDistBest = minDist;
        labelBest = label;
        currentMedoidsBest = currentMedoids;
     end
end

%% Once the medoids are defined, store the outputs
weights = zeros(nbclusters,1);
for i = 1:nbclusters
    weights(i) = sum(labelBest == i);
end

Clustering.label = labelBest;
Clustering.medoids = currentMedoidsBest;
Clustering.weights = weights;

end
