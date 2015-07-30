classdef bsttree_pq < handle
    
    properties
        data % array of points for our purpose
        sep = 0 % cluster centers
        nsize = 0 % number of elements in the node
        ndepth = 0 % depth of the node
        
    end
    
    properties  (SetAccess = private)
        left = bsttree_pq.empty; % left subtree
        right = bsttree_pq.empty; % right subtree
    end
    
    methods
        
        % initialization
        % data:    data from which the tree needs to be constructed
        % msize:   maximum number of data points allowed in a node
        % mdepth:  maximum depth of the tree
        % kernelf: kernel function used as the distance metric
        % depth:   intial depth of the tree
        function root = bsttree_pq(data, msize, mdepth, kernelf, depth)
            % initialize the root
            root.data = data;
            sizep = size(data);
            sizep = sizep(2);
            root.nsize = sizep;
            root.ndepth = depth;
            
            % check that nsize and ndepth are in acceptable range
            if(root.nsize <= msize || root.ndepth >= mdepth)
                return;
                
                % run kernel k-means and split the data
            else
                % get classification and medians
                [labels,medoids] = kkmeans_m(data, kernelf, 2, root.nsize);
                
                % sort the data by labels
                [m,ind] = sort(labels);
                
                % assign points with label 1 to left node and label 2 to
                % right node
                num1 = sum(labels == 1);
                lchild = data(:,ind(1:num1));
                rchild = data(:, ind(num1 + 1:end));
                
                % store medians in sep
                root.sep = data(:,medoids);
                
                % recursively call bstrree on left and right node
                root.left = bsttree_pq(lchild, msize, mdepth, kernelf, depth + 1);
                root.right = bsttree_pq(rchild, msize, mdepth, kernelf, depth + 1);
            end % end if
        end % end function
        
    end % end methods
    
    methods
        
        % Priority queue based search for NN. Described in FLANN.
        % root:    the tree containing data
        % query:   query point whose NN are to be searched
        % kernelf: kernel function used as distance metric
        function points = psearch(root, query, kernelf, max)           
           % call another function to traverse the tree
           r = travtree(root, query, kernelf);
           
           % select first max number of points
           points = r(:,1:max);        
        end
        
        % makes a priority queue based structure while traversing the tree
        % root:    the tree containing data
        % query:   query point whose NN are to be searched
        % kernelf: kernel function used as distance metric
        function q = travtree(root, query, kernelf)
            % base case
            % if the root is a leaf            
            if(isempty(root.left))
               q = root.data;
               return;
            end
            
            % get the cluster centers
            med1 = root.sep(:,1);
            med2 = root.sep(:,2);
            
            % calculate distance between query point and cluster centers
            distl = distk(query, med1, kernelf);
            distr = distk(query, med2, kernelf);
            
            % recursive call to whichever center is closer 
            if(distl < distr)
                % store data according to distance from query point
                q = horzcat(travtree(root.left, query, kernelf),...
                    root.right.data);
            else
                % store data according to distance from query point
                q = horzcat(travtree(root.right, query, kernelf),...
                    root.left.data);
            end % end if
        end % end function

    end % end methods
  
end % end class