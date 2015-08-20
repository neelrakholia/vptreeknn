function randomvp(qpoints, dbpoints, nquery, npoints, k, nt, sig, ...
    mpnts, mlvl, npiece, qlab, dblab)
%--------------------------------------------------------------------------
% RANDOMVP Perform random vp tree nn search on given data for an rbf kernel
% distance metric
%   Input
%       qpoints - the query data
%       dbpoints - the database points
%       nquery - number of query points
%       npoints - number of database points
%       k - number of nearest neighbors
%       nt - number of trees to construct
%       sig - kernel bandwidth
%       mpnts - limit on the number of points allowed in the leaf
%       mlvl - limit on the depth of the tree
%       npiece - number of partitions for quadratic search
%       qlab - labels for query points
%       dblab - labels for database points
%
%   Ouput
%       Information about the run in printed
%--------------------------------------------------------------------------

% define test and train sets
test = qpoints;
train = dbpoints;

% set number of queries and points
m = nquery;
n = npoints;

% number of nearest neighbors, and number of trees to generate
K = k;
ntree = nt;

% example kernel
sigma = sig;

% tree options
maxPointsPerNode = mpnts;
maxLevel        =  mlvl;

% brute force search
tic
piece = npiece;
actual_nn = zeros(m,K);
actual_nn(1:m/piece, :) = kknn(train,1:n,test(:,1:m/piece),sigma,K,n);
for i = 2:piece
    actual_nn((i - 1)*(m/piece) + 1:i*m/piece,:) = ...
        kknn(train,1:n,test(:,(i - 1)*(m/piece) + 1:i*m/piece),sigma,K,n);
end
toc

% construct tree and search for nearest neighbors
points = zeros(m,K);

% storing previous iteration
test_nn = ones(m, K);

% array for storing time
elapsed_time_array = [];

% search for neighbors for each query point
disteval = 0;
treeeval = 0;
k = 0;
acc = 0;

% iterate through all the the trees
while(k <= ntree && acc < 0.9)
    % construct tee
    tic
    root = bsttree_vp(train, 1:n, maxPointsPerNode, maxLevel, sigma, 0, 0);
    elapsed_time_array(end + 1) = toc;
    
    % search tree
    tic
    [new_nn,deval] = travtree2n(root, test, sigma, ...
        1:m, train, K, points, test_nn, 0);
    test_nn = new_nn;
    points = test_nn;
    disteval = disteval + deval;
    elapsed_time_array(end + 1) = toc;
    
    % evaluate performace
    tic
    treeeval = treeeval + root.dise;
    
    suml = 0;
    % calculate accuracy by comparing neighbors found
    if(K == 1)
        for i = 1:m
            suml = suml + length(intersect(points(i), actual_nn(i)));
        end
    else
        for i = 1:m
            suml = suml + length(intersect(points(i,:), actual_nn(i,:)));
        end
    end
    
    % calculate distance ratio
    distr = zeros(m, 1);
    for i = 1:m
        dist_actual = distk(train(:,actual_nn(i,:)),test(:,i),sigma);
        dist_app = distk(train(:,points(i,:)),test(:,i),sigma);
        distr(i) = mean(dist_app ./ dist_actual); 
    end
    
    % print accuracy
    acc = suml/(m*K);
    fprintf('Accuracy: %f\n', suml/(m*K));
    
    if(nargin > 10)
       fprintf('Naive classification accuracy for tree NN: %f\n',...
           sum(dblab(points(:,1)) == qlab)/nquery);
       fprintf('Naive classification accuracy for quad search: %f\n',...
           sum(dblab(actual_nn(:,1)) == qlab)/nquery);
       fprintf('Classification accuracy for tree NN: %f\n',...
           sum(mode(dblab(points),2) == qlab)/nquery);
       fprintf('Classification accuracy for quad search: %f\n',...
           sum(mode(dblab(actual_nn),2) == qlab)/nquery);
    end
    
    % display fraction of distance evaluations conducted while constructing
    % the tree
    fprintf('Distance evaluations in tree: %f\n', treeeval/(m*n));
    
    % display fraction of distance evaluations conducted
    fprintf('Distance evaluations: %f\n', disteval/(m*n));
    
    % display total number of distance evaluations:
    fprintf('Total distance evaluations: %f\n', disteval/(m*n) + ...
        treeeval/(m*n));
    
    % print ratio of distance
    fprintf('Average ratio of distance: %f\n',mean(distr));
    
    k = k + 1;
    elapsed_time_array(end + 1) = toc;
end

% display important run info
fprintf('Time spent constructing trees: %g\n',...
    sum(elapsed_time_array(1:3:end)));
fprintf('Time spent searching trees: %g\n',...
    sum(elapsed_time_array(2:3:end)));
fprintf('Total time spent running algo: %g\n',...
    sum(elapsed_time_array(1:3:end)) + sum(elapsed_time_array(2:3:end)));
fprintf('Total time spent running: %g\n',...
    sum(elapsed_time_array))

end