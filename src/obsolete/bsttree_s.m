classdef bsttree_s < handle
    
    properties
        data % array of points for our purpose
        nsize = 0 % number of elements in the node
        ndepth = 0 % depth of the node
    end
    
    properties  (SetAccess = private)
        left = bsttree_s.empty; % left subtree
        right = bsttree_s.empty; % right subtree
    end
    
    methods
        
        % initialization
        % data:    data from which the tree needs to be constructed
        % msize:   maximum number of data points allowed in a node
        % mdepth:  maximum depth of the tree
        % kernelf: kernel function used as the distance metric
        % depth:   intial depth of the tree
        function root = bsttree_s(data, msize, mdepth, kernelf, depth, ap)
            if(nargin == 5)
                % initialize the root
                root.data = data;
                sizep = size(data);
                sizep = sizep(2);
                root.nsize = sizep;
                root.ndepth = depth;
                
                % check that nsize and ndepth are in acceptable range
                if(root.nsize <= msize || root.ndepth >= mdepth)
                    return;
                    
                % run kernel k-means
                else
                    % get classification
                    labels = kkmeans_ap(data, kernelf, 2, root.nsize);
                    
                    % sort the data by labels
                    [m,ind] = sort(labels);
                    
                    % assign points with label 1 to left node and label 2 to
                    % right node
                    num1 = sum(labels == 1);
                    lchild = data(:,ind(1:num1));
                    rchild = data(:, ind(num1 + 1:end));
                    
                    % recursively call bstrree on left and right node
                    root.left = bsttree_s(lchild, msize, mdepth, ...
                        kernelf, depth + 1);
                    root.right = bsttree_s(rchild, msize, mdepth, ...
                        kernelf, depth + 1);
                end % end if
            
            % construct naive tree
            else
                % initialize the root
                root.data = data;
                sizep = size(data);
                sizep = sizep(2);
                root.nsize = sizep;
                root.ndepth = depth;
                
                % check that nsize and ndepth are in acceptable range
                if(root.nsize <= msize || root.ndepth >= mdepth)
                    return;
                    
                % run kernel k-means
                else
                    % get classification
                    % randomly select 2 data points
                    n = root.data(:, randperm(root.nsize));
                    n1 = n(:,1);
                    n2 = n(:,2);
                    
                    % get labels
                    labels = zeros(root.nsize, 1);
                    
                    for i = 1:root.nsize
                       distl = distk(n1, root.data(:,i), kernelf);
                       distr = distk(n2, root.data(:,i), kernelf);
                       
                       if(distl < distr)
                           labels(i) = 1;
                       else
                           labels(i) = 2;
                       end
                    end
                    
                    % sort the data by labels
                    [m,ind] = sort(labels);
                    
                    % assign points with label 1 to left node and label 2 to
                    % right node
                    num1 = sum(labels == 1);
                    lchild = data(:,ind(1:num1));
                    rchild = data(:, ind(num1 + 1:end));
                    
                    % recursively call bstrree on left and right node
                    root.left = bsttree_s(lchild, msize, mdepth, ...
                        kernelf, depth + 1, ap);
                    root.right = bsttree_s(rchild, msize, mdepth, ...
                        kernelf, depth + 1, ap);
                end % end if
            end % end if else
        end % end function
        
    end % end method
    
    methods
        
        % search tree for k nearest neighbors for a query point
        % root:     the tree
        % query:    the point whose nearest neighbors we want to compute
        % kernelf:  the kernel function used as distance metric
        % k:        number of nearest neighbors we want to get
        function points = sampsearch(root, query, kernelf, k, ns)
            % if the root is empty
            if(root == 0)
                points = 0;
                return;
            end
            
            % if the root is a leaf
            if(isempty(root.left))
               points = root.data;
               return;
            end
            
            % if root data size is smaller than data size
            if(root.nsize < ns)
                points = root.data;
                return;
            end
            
            % if one of the children has data size smaller than sample size
            if(root.left.nsize < ns && root.right.nsize >= ns)
                point1 = root.left.data;
                point2 = sampsearch(root.right, query, kernelf, k, ns);
                points = horzcat(point1, point2);
                return;
            end
            
            % if one of the children has data size smaller than sample size            
            if(root.right.nsize < ns && root.left.nsize >= ns)
                point1 = root.right.data;
                point2 = sampsearch(root.left, query, kernelf, k, ns);
                points = horzcat(point1, point2);
                return;
            end
            
            % if both children are sufficiently large
            
            % randomly select ns number of data points from both nodes
            n1 = root.left.data(:, randperm(root.left.nsize));
            n2 = root.right.data(:, randperm(root.right.nsize));
            n1 = n1(:, 1:ns);
            n2 = n2(:, 1:ns);
            
            % select the data points that is closest to the query point in
            % both the nodes
            n1 = kknn(n1, query, kernelf, 1, ...
                ns);
            n2 = kknn(n2, query, kernelf, 1, ...
                ns);
            
            % calculate the distance from the query point to the selected
            % data points
            distl = distk(n1, query, kernelf);
            distr = distk(n2, query, kernelf);
            
            % if point is sufficiently far from right
            if(distl < distr)
               points = sampsearch(root.left, query, kernelf, k, ns);
               return;
            % if point is sufficiently far from left
            else
               points = sampsearch(root.right, query, kernelf, k, ns);
               return;
            end % end if
        end % end function

    end % end methods
 
end % end class