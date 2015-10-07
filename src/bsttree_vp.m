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
        function root = bsttree_vp(data, indi, msize, mdepth, kernel, ...
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
            
            % check that nsize and ndepth are in acceptable range
            if(root.nsize <= msize || root.ndepth >= mdepth)
                return;
                
            % get left and right classification
            else
                % get classification and radius
                [datal, datar, indl, indr, radi, cen, diseval] = ...
                    classify_vp(data, ...
                    indi, root.nsize, kernel, diseval);
                
                % update disteval
                root.dise = diseval;
                
                % assign vantage point
                root.cent = cen;
                
                % assign points
                lchild = datal;
                rchild = datar;
                
                % store radius
                root.rad = radi;
                
                % recursively call bstrree on left and right node
                root.left = bsttree_vp(lchild, indl, msize, mdepth, ...
                    kernel, depth + 1, diseval);
                root.right = bsttree_vp(rchild, indr, msize, mdepth, ...
                    kernel, depth + 1, diseval);
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
        
        function [nn,dev] = travtree2n(root, query, global_id, data,...
                k, nn, prev, dev)
        %------------------------------------------------------------------
        % TRAVTREE2N Random greedy search for NN using vp-trees
        %   Input 
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
            
            % base case
            % if the root is a leaf
            if(isempty(root.left))
                % search for nn
                prev_id = reshape(prev(global_id,:),1,k*numel(global_id));
                search_id = unique([root.ind, prev_id]);
                q = kknn(data, search_id, query, root.kernel, k, numel(search_id));

                % store nn
                nn(global_id,:) = q;
                
                % update computations
                dev = dev + numel(search_id)*numel(global_id);
                return;
            end
            
            % get the radius and center
            radius = root.rad;
            center = root.cent;
            
            % calculate distance between query point and center
            dist = root.kernel(query, center);
            larr = dist < radius;
            
            % get the right and left queries
            indl = find(larr);
            indr = find(~larr);
            
            % recursive call to whichever center is closer
            if(numel(indl) > 0)
                % store data according to distance from query point
                [nn,dev] = travtree2n(root.left, query(:,indl), ...
                    global_id(indl), data, k, nn, prev, dev);
            end
            if(numel(indr) > 0)
                % store data according to distance from query point
                [nn,dev] = travtree2n(root.right, query(:, indr), ...
                    global_id(indr), data, k, nn, prev, dev);
            end % end if
        end % end function
        
        
        
        function [nn, dev] = PartialBacktracking(root, queries, global_ids, data, k, nn, prev, dev, num_backtracks)
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
            
    
    
            num_queries = size(queries, 2);

            backtracks_in = ones(num_queries,1) * -1;

            backtrack_queues = cell(num_queries, 1); % cell array of java priority queues -- used for partial backtrackin search

            % Some visualization that only works for d = 2
            do_plot = false;

            if (do_plot)
                plot_colors = {'r', 'b', 'g', 'k'};
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
            
            back_ind = 0;
            
            fprintf('Initial call (no backtracks)\n');
            % do the initial call
            [nn, dev] = PartialBacktrackingHelper(root, queries, global_ids, data, k, nn, prev, dev, ...
                backtracks_in, true);        

            backtracks = -1 * ones(num_queries, num_backtracks);
            % now, sort them
            for q_ind = 1:num_queries
               [~, inds] = sort(backtrack_queues{q_ind}(:,2));
               backtracks(q_ind, :) = backtrack_queues{q_ind}(inds(1:num_backtracks), 1);
            end
            
            
            % now we'll repeat the search, but with the alternative path
            for back_ind = 1:num_backtracks

                prev = nn;
                
                 fprintf('Backtracks no: %d\n', back_ind);
                
                [nn, dev] = PartialBacktrackingHelper(root, queries, global_ids, data, k, nn, prev, dev, ...
                    backtracks(:,back_ind), false);        
               
                
            end 
            
        
        % helper for the above, handles the recursion
        % backtracks are stored as a level (at which we go the other way)
        % for each query
        function [nn, dev] = PartialBacktrackingHelper(root, queries, ...
                global_id, data, k, nn, prev, dev, backtracks_in, store_backtracks)
            
            % base case
            % if the root is a leaf
            if(isempty(root.left))
                % search for nn
                prev_id = reshape(prev(global_id,:),1,k*numel(global_id));
                search_id = unique([root.ind, prev_id]);
                q = kknn(data, search_id, queries, root.kernel, k, numel(search_id));

                
                if (do_plot && ~isempty(find(global_id == 1,1)))
                    scatter(root.data(1,:), root.data(2,:), [], plot_colors{back_ind+1});
                end                

                
                % store nn
                nn(global_id,:) = q;
                
                % update computations -- 
                % TODO: don't want to repeat kernel evaluations here
                dev = dev + numel(search_id)*numel(global_id);

                return;
                
            end
            
            % get the radius and center
            radius = root.rad;
            center = root.cent;
            
            % calculate distance between query point and center
            dist = root.kernel(queries, center);
            larr = dist < radius;

            % update the backtrack checking for the next iteration
            if (store_backtracks)
                these_dists = abs(dist - radius);
                for i = 1:numel(these_dists)
%                     fprintf('Adding to queue: [%d, %g]\n', root.ndepth, these_dists(i));
                    backtrack_queues{global_id(i)}(root.ndepth+1, :) = [root.ndepth, these_dists(i)];
                end
            end
            
  
            % now we need to update these with backtracks
            % just flip the bit of anyone who needs to backtrack here
            larr(backtracks_in == root.ndepth) = ~larr(backtracks_in == root.ndepth);
            
            % get the right and left queries
            indl = find(larr);
            indr = find(~larr);
           
%             fprintf('Call at node level: %d\n', root.ndepth);
%             global_id
%             indl
%             indr
%             backtracks_in
%             fprintf('\n\n\n');

            
            % recursive call to whichever center is closer
            if(numel(indl) > 0)
                % store data according to distance from query point
                [nn,dev] = PartialBacktrackingHelper(root.left, queries(:,indl), ...
                    global_id(indl), data, k, nn, prev, dev, backtracks_in(indl), store_backtracks);
            end
            if(numel(indr) > 0)
                % store data according to distance from query point
                [nn,dev] = PartialBacktrackingHelper(root.right, queries(:, indr), ...
                    global_id(indr), data, k, nn, prev, dev, backtracks_in(indr), store_backtracks);
            end % end if
            
            
        end % helper function
        
                    
        end % function
        

        
    end % end methods
    
end % end class











