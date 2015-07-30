% source: https://sites.google.com/site/radhacr/academics/projects/software
% "Approximate Kernel k-means: Solution to Large Scale Kernel Clustering"
% Krect - m X N rectangular kernel matrix  
% k - number of clusters
% max_iter - maximum number of iterations
% indices - indices of the sampled data points
function label=approx_kkmeans(Krect,k,max_iter,indices)   

    [m,N] = size(Krect);
    Khat = Krect(1:m,indices);
    Krect = full(Krect);
    
    Khat_inv = pinv(full(Khat));
    T = Khat_inv*Krect;
    
    init_labels = ceil(rand(1,N)*k);
    label = init_labels;
    
    last = 0;
    t = 0;
    
    while(any(label~=last) && t < max_iter)
	    % compute sparse membership
	    U = sparse(label,1:N,1,k,N,N);
	    U = full(bsxfun(@rdivide,U,sum(U,2)));
        % solve for cluster center weights
	    alpha = (T*U')';  
        % distance between cluster center and each object
	    D = bsxfun(@plus,-2*alpha*Krect,diag(alpha*Khat*alpha')); 
	    last = label;
        % assign new labels
	    [~,label] = min(D); 
	        
	    t = t + 1;
    end % end while
    
end % end function
