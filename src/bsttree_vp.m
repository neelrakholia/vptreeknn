classdef bsttree_vp < handle
%--------------------------------------------------------------------------
% BSTTREE_VP Constructs a random vantage point tree
%
% IMPORTANT: we're going to consistently search for the MAXIMUM similarity.
% So, for kernel distances, we'll need to flip the sign.
%--------------------------------------------------------------------------    
    properties
        data % array of points for our purpose
        ind % indices of data points
        rad = 0 % radius of inner cluster
        cent % point chosen as vantage point
        nsize = 0 % number of elements in the node
        ndepth = 0 % depth of the node
        dise = 0 % number of distance evaluations
        kernel % handle for kernel function evaluations
        kernel_dist % handle for the distance version of the kernel function

    end
    
    properties  (SetAccess = private)
        left = bsttree_vp.empty; % left subtree
        right = bsttree_vp.empty; % right subtree
    end
    
    methods
        
        % initialization
        % data:    data from which the tree needs to be constructed
        % msize:   maximum number of data points allowed in a node
        % mdepth:  maximum depth of the tree
        % kernelf: rbf kernel bandwidth
        % depth:   intial depth of the tree
        % diseval: number of distance evaluations
        function root = bsttree_vp(data, indi, msize, mdepth, kernel, kernel_dist, ...
            depth, diseval)
        %------------------------------------------------------------------
        % BSTTREE_VP Initializes a vp-tree based on the given data
        %   Input 
        %       data - database points to be organized into a vp-tree
        %       indi - global ids of database points
        %       msize - maximum points per leaf
        %       mdepth - maximum allowed tree depth
        %       kernel - handle for kernel function
        %       depth - initial depth of the tree
        %       diseval - variable to keep track of distance evaluations
        %
        %   Output 
        %       root - pointer to the root of the tree
        %------------------------------------------------------------------
            
            % initialize the root
            root.data = data;
            sizep = size(data, 2);
            root.nsize = sizep;
            root.ndepth = depth;
            root.ind = indi;
            root.kernel = kernel;
            root.kernel_dist = kernel_dist;
        
global do_plot
            

            % check that nsize and ndepth are in acceptable range
            if(root.nsize <= msize || root.ndepth >= mdepth)

                % we'll define the VP of a leaf node to be a random point
                % in the node -- used for determining the search direction 
                root.cent = data(:, randi(root.nsize, 1));
                
                return;
                
            % get left and right classification
            else
                % get classification and radius
%                 [datal, datar, indl, indr, radi, cen, diseval] = ...
%                     classify_vp(data, ...
%                     indi, root.nsize, kernel_dist, diseval);
                

                [datal, datar, indl, indr, radi, cen, diseval] = ...
                    classify_vp(data, ...
                    indi, root.nsize, kernel, diseval);

                % update disteval
                root.dise = diseval;
                
                % assign vantage point
                root.cent = cen;

                if (do_plot)
                    plot_colors = {'r', 'b', 'g', 'k'};
                    scatter(cen(1), cen(2), 200, plot_colors{depth+1}, 's');
                end
    
                % assign points
                lchild = datal;
                rchild = datar;
                
                % store radius
                root.rad = radi;
                
                % recursively call bstrree on left and right node
                root.left = bsttree_vp(lchild, indl, msize, mdepth, ...
                    kernel, kernel_dist, depth + 1, diseval);
                root.right = bsttree_vp(rchild, indr, msize, mdepth, ...
                    kernel, kernel_dist, depth + 1, diseval);
            end % end if
        end % end function
        
    end % end methods
    
    methods
        
        function [points,deval] = psearch(root, data, query, k)
        %------------------------------------------------------------------
        % PSEARCH Priority queue based search for NN. Described in FLANN.
        %   Input 
        %       root - pointer to tree root
        %       data - database points to be organized into a vp-tree
        %       query - query points whose NN are to be determined
        %       sigma - kernel bandwidth parameter
        %       k - number of nearest neighbors to be found
        %
        %   Output 
        %       points - nearest neighbors
        %       deval - total number of distance evaluations in the search
        %------------------------------------------------------------------
            
            % initialize dk as infinity
            dk = -inf;
            deval = 0;
            
            % call another function to traverse the tree
            [r,~,deval] = travtree(root, data, query, k, dk, [], deval);
            
            % select first max number of points
            points = r;
        end
        
        function [q, dk, deval] = travtree(root, data, query, ...
                k, dk, q_in, deval)
        %------------------------------------------------------------------
        % TRAVTREE Priority queue based search for NN. Described in FLANN.
        %   Input 
        %       root - pointer to tree root
        %       data - database points to be organized into a vp-tree
        %       query - query points whose NN are to be determined
        %       k - number of nearest neighbors to be found
        %       dk - distance to furthest neighbor
        %       q_in - array for storing found neighbors
        %       deval - keeping track of distance evaluations
        %
        %   Output 
        %       q - nearest neighbors
        %       dk - distance to the furthest nearest neighbor
        %       deval - total number of distance evaluations in the search
        %------------------------------------------------------------------
            % base case
            % if the root is a leaf
            if(isempty(root.left))
                % search for optimal neighbors 
                indi = root.ind;
                q = kknn(data, [indi, q_in], query, root.kernel, k, ...
                    numel(indi)+numel(q_in));
                
                % update distance evaluations
                deval = deval + numel(indi)+numel(q_in);
                
                % get current NN
                dk_curr = root.kernel(query, data(:, q(k)));

                % uodate them if required
                if(dk_curr > dk)
                    dk = dk_curr;
                end
                
                return; 
            end
            
            % get the radius and center
            radius = root.rad;
            center = root.cent;
            
            % calculate distance between query point and center
            dist = root.kernel(query, center);

            if(dist < radius)
                % store data according to distance from query point
                [q, dk, deval] = travtree(root.left, data, query,...
                    k, dk, q_in, deval);
                
                % if right cannot be pruned
                if(dist + dk > radius)
                    [q, dk, deval] = travtree(root.right, data, query, ...
                        k, dk, q, deval);
                end
            else 
                % store data according to distance from query point
                [q, dk, deval] = travtree(root.right, data, query,...
                    k, dk, q_in, deval);
                
                % if left cannot be pruned
                if(dist < radius + dk)
                    [q, dk, deval] = travtree(root.left, data, ...
                        sigma, k, dk, q, deval);
                end
            end
        end % end function
        
        function [nn_inds, nn_dists, dev] = travtree2n(root, query, data,...
                k, prev_ids, prev_dists, dev)
        %------------------------------------------------------------------
        % TRAVTREE2N Random greedy search for NN using vp-trees
        %   Input 
        %       root - pointer to tree root
        %       query - query points whose NN are to be determined
        %       global_id - global ids of data points
        %       query_ids -- ids of query points 
        %       data - database points to be organized into a vp-tree
        %       k - number of nearest neighbors to be found
        %       nn - updated matrix of nn
        %       prev_ids - previous matrix of nn indices
        %       prev_dists - previous matrix of nn distances
        %       dev - keeping track of distance evaluations
        %
        %   Output 
        %       nn - nearest neighbors for current iteration
        %       dev - total number of distance evaluations in the search
        %------------------------------------------------------------------
  
        global do_plot

          
            % Some visualization that only works for d = 2
            
            if (do_plot)
                plot_colors = {'r', 'b', 'g', 'k'};
                scatter(root.data(1,:), root.data(2,:), [], plot_colors{root.ndepth+1});
            end
            

            % base case
            % if the root is a leaf
            if(isempty(root.left))
                % search for nn
                kvals = root.kernel(query, data(:,root.ind));
                [nn_dists, nn_inds] = knn_update([prev_dists, kvals], [prev_ids, repmat(root.ind, size(prev_ids,1),1)], k);
                
                % update computations
                dev = dev + size(query,2)*numel(root.ind);
                
                return;
            end

            % we'll look at the VPs of the children and go to the closer
            % one
            left_vp = root.left.cent;
            right_vp = root.right.cent;
            
            kvals = root.kernel(query, [left_vp, right_vp]);
            larr = kvals(:,1) > kvals(:,2);
            
            dev = dev + 2 * size(query,2);

            % get the right and left queries
            indl = find(larr);
            indr = find(~larr);
            
            nn_inds = zeros(size(query,2),k);
            nn_dists = zeros(size(query,2), k);
            
            % recursive call to whichever center is closer
            if(numel(indl) > 0)
                % store data according to distance from query point
                [nn_inds(indl,:), nn_dists(indl,:), dev] = travtree2n(root.left, query(:,indl), ...
                    data, k, prev_ids(indl,:), prev_dists(indl,:), dev);
            end
            if(numel(indr) > 0)
                % store data according to distance from query point
                [nn_inds(indr,:), nn_dists(indr,:), dev] = travtree2n(root.right, query(:,indr), ...
                    data, k, prev_ids(indr,:), prev_dists(indr,:), dev);
            end % end if
            
            
        end % end function
        
        
        function [nn_inds, nn_dists, dev] = PartialBacktracking(root, queries, data,...
                k, prev_nn_inds, prev_nn_dists, num_backtracks)

        %------------------------------------------------------------------
        % PartialBacktraking Random greedy search for NN using vp-trees
        %  This method does some partial backtracking.   
        % Input 
        %       root - pointer to tree root
        %       query - query points whose NN are to be determined
        %       global_id - global ids of data points
        %       data - database points to be organized into a vp-tree
        %       k - number of nearest neighbors to be found
        %       nn - updated matrix of nn
        %       prev - previous matrix of nn
        %       dev - keeping track of distance evaluations
        %
        %   Output 
        %       nn - nearest neighbors for current iteration
        %       dev - total number of distance evaluations in the search
        %------------------------------------------------------------------

                  global do_plot
                  
            dev = 0;

            num_queries = size(queries, 2);

            backtracks_in = ones(num_queries,1) * -1;

            backtrack_queues = cell(num_queries, 1); % cell array of java priority queues -- used for partial backtrackin search

            % Some visualization that only works for d = 2
            
            if (do_plot)
%                 plot_colors = {'r', 'b', 'g', 'k'};
                figure()
                scatter(queries(1,1), queries(2,1), 200, 'm', 'x');
                hold on;    
            end
            
            % just inserting this here so we don't have to track another 
            % parameter
            max_depth = 0;
            node = root;
            while (~isempty(node.left))
               node = node.left;
               max_depth = max_depth + 1;
            end
            
            for q_ind = 1:num_queries
               backtrack_queues{q_ind} = 10^10 * ones(max_depth, 2);
            end
            
            fprintf('Initial call (no backtracks)\n');
            % do the initial call
            [nn_inds, nn_dists, dev] = PartialBacktrackingHelper(root, queries, data, k, prev_nn_inds, prev_nn_dists, dev, ...
                backtracks_in, true, 1:num_queries);        

            backtracks = -1 * ones(num_queries, num_backtracks);
            % now, sort them
            for q_ind = 1:num_queries
               [~, inds] = sort(backtrack_queues{q_ind}(:,2));
               backtracks(q_ind, :) = backtrack_queues{q_ind}(inds(1:num_backtracks), 1);
            end
            
            % now we'll repeat the search, but with the alternative path
            for back_ind = 1:num_backtracks

                 fprintf('Backtracks no: %d\n', back_ind);
                prev_nn_ids = nn_inds;
                prev_nn_dists = nn_dists;
                 
                [nn_inds, nn_dists, dev] = PartialBacktrackingHelper(root, queries, data, k, prev_nn_ids, prev_nn_dists, ...
                    dev, backtracks(:,back_ind), false, 1:num_queries);        
               
                
            end 
            
        
        % helper for the above, handles the recursion
        % backtracks are stored as a level (at which we go the other way)
        % for each query
       
        function [nn_inds, nn_dists, dev] = PartialBacktrackingHelper(root, query, data,...
                k, prev_ids, prev_dists, dev, backtracks_in, store_backtracks, query_global_ids)

            
            % base case
            % if the root is a leaf
            if(isempty(root.left))
                % search for nn
                kvals = root.kernel(queries, data(:,root.ind));
                [nn_dists, nn_inds] = knn_update([prev_dists, kvals], [prev_ids, repmat(root.ind, size(prev_ids,1),1)], k);
                
                % update computations
                dev = dev + size(query,2)*numel(root.ind);
                
                return;
            end

            % we'll look at the VPs of the children and go to the closer
            % one
            left_vp = root.left.cent;
            right_vp = root.right.cent;
            
            kvals = root.kernel(queries, [left_vp, right_vp]);
            larr = kvals(:,1) > kvals(:,2);

            % update the backtrack checking for the next iteration
            if (store_backtracks)
                these_dists = abs(kvals(:,1) - kvals(:,2));
                for i = 1:numel(these_dists)
                    backtrack_queues{query_global_ids(i)}(root.ndepth+1, :) = [root.ndepth, these_dists(i)];
                end
            end
            
            larr(backtracks_in == root.ndepth) = ~larr(backtracks_in == root.ndepth);
            
            % get the right and left queries
            indl = find(larr);
            indr = find(~larr);
            
            nn_inds = zeros(size(query,2),k);
            nn_dists = zeros(size(query,2), k);
            
            % recursive call to whichever center is closer
            if(numel(indl) > 0)
                % store data according to distance from query point
                [nn_inds(indl,:), nn_dists(indl,:), dev] = travtree2n(root.left, query(:,indl), ...
                    data, k, prev_ids(indl,:), prev_dists(indl,:), dev);
            end
            if(numel(indr) > 0)
                % store data according to distance from query point
                [nn_inds(indr,:), nn_dists(indr,:), dev] = travtree2n(root.right, query(:,indr), ...
                    data, k, prev_ids(indr,:), prev_dists(indr,:), dev);
            end % end if
            
            
        end % helper function
        
                    
        end % function
        

        
    end % end methods
    
end % end class











