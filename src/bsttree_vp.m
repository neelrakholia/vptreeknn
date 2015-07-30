classdef bsttree_vp < handle
%--------------------------------------------------------------------------
% BSTTREE_VP Constructs a random vantage point tree
%--------------------------------------------------------------------------    
    properties
        data % array of points for our purpose
        ind % indices of data points
        rad = 0 % radius of inner cluster
        cent % point chosen as vantage point
        nsize = 0 % number of elements in the node
        ndepth = 0 % depth of the node
        dise = 0 % number of distance evaluations
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
        function root = bsttree_vp(data, indi, msize, mdepth, sigma, ...
            depth, diseval)
        %------------------------------------------------------------------
        % BSTTREE_VP Initializes a vp-tree based on the given data
        %   Input 
        %       data - database points to be organized into a vp-tree
        %       indi - global ids of database points
        %       msize - maximum points per leaf
        %       mdepth - maximum allowed tree depth
        %       sigma - kernel bandwidth parameter
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
            
            % check that nsize and ndepth are in acceptable range
            if(root.nsize <= msize || root.ndepth >= mdepth)
                return;
                
            % get left and right classification
            else
                % get classification and radius
                [datal, datar, indl, indr, radi, cen, diseval] = ...
                    classify_vp(data, ...
                    indi, root.nsize, sigma, diseval);
                
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
                    sigma, depth + 1, diseval);
                root.right = bsttree_vp(rchild, indr, msize, mdepth, ...
                    sigma, depth + 1, diseval);
            end % end if
        end % end function
        
    end % end methods
    
    methods
        
        function [points,deval] = psearch(root, data, query, sigma, k)
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
            dk = inf;
            deval = 0;
            
            % call another function to traverse the tree
            [r,~,deval] = travtree(root, data, query, sigma, k, dk, [], deval);
            
            % select first max number of points
            points = r;
        end
        
        function [q, dk, deval] = travtree(root, data, query, sigma, ...
                k, dk, q_in, deval)
        %------------------------------------------------------------------
        % TRAVTREE Priority queue based search for NN. Described in FLANN.
        %   Input 
        %       root - pointer to tree root
        %       data - database points to be organized into a vp-tree
        %       query - query points whose NN are to be determined
        %       sigma - kernel bandwidth parameter
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
                q = kknn(data, [indi, q_in], query, sigma, k, ...
                    numel(indi)+numel(q_in));
                
                % update distance evaluations
                deval = deval + numel(indi)+numel(q_in);
                
                % get current NN
                dk_curr = distk(query, data(:,q(k)), sigma);
                
                % uodate them if required
                if(dk_curr < dk)
                    dk = dk_curr;
                end
                
                return; 
            end
            
            % get the radius and center
            radius = root.rad;
            center = root.cent;
            
            % calculate distance between query point and center
            dist = distk(query, center, sigma);

            if(dist < radius)
                % store data according to distance from query point
                [q, dk, deval] = travtree(root.left, data, query, sigma,...
                    k, dk, q_in, deval);
                
                % if right cannot be pruned
                if(dist + dk > radius)
                    [q, dk, deval] = travtree(root.right, data, query, ...
                        sigma, k, dk, q, deval);
                end
            else 
                % store data according to distance from query point
                [q, dk, deval] = travtree(root.right, data, query, sigma,...
                    k, dk, q_in, deval);
                
                % if left cannot be pruned
                if(dist < radius + dk)
                    [q, dk, deval] = travtree(root.left, data, query, ...
                        sigma, k, dk, q, deval);
                end
            end
        end % end function
        
        function [nn,dev] = travtree2n(root, query, sigma, global_id, data,...
                k, nn, prev, dev)
        %------------------------------------------------------------------
        % TRAVTREE2N Random greedy search for NN using vp-trees
        %   Input 
        %       root - pointer to tree root
        %       query - query points whose NN are to be determined
        %       sigma - kernel bandwidth parameter
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
                q = kknn(data, search_id, query, sigma, k, numel(search_id));

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
            dist = distk(query, center, sigma);
            larr = dist < radius;
            
            % get the right and left queries
            indl = find(larr);
            indr = find(~larr);
            
            % recursive call to whichever center is closer
            if(numel(indl) > 0)
                % store data according to distance from query point
                [nn,dev] = travtree2n(root.left, query(:,indl), sigma, ...
                    global_id(indl), data, k, nn, prev, dev);
            end
            if(numel(indr) > 0)
                % store data according to distance from query point
                [nn,dev] = travtree2n(root.right, query(:, indr), sigma, ...
                    global_id(indr), data, k, nn, prev, dev);
            end % end if
        end % end function
        
    end % end methods
    
end % end class