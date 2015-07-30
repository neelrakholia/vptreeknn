function k = ssk_dyn(s1, s2, sublen, lambda)
%--------------------------------------------------------------------------
%SSK_DYN Computes ssk using dynamic programming approach
%   Input
%       s1 - string 1
%       s2 - string 2
%       sublen - subsequence length
%       lambda - tuning parameter
%
%   Output
%       k - kernel value
%--------------------------------------------------------------------------

% calculate the lengths
len1 = length(s1);
len2 = length(s2);

% initialize array
Kp = zeros(len1 + 1, len2 + 1, sublen + 1);

% base case
Kp(:,1,:) = 0;
Kp(1,:,:) = 0;
Kp(:,:,1) = 1;

% loop for subsequence length
for i = 2:sublen+1
    
    % loop over first string length
    for s_ind = 2:len1+1
        
        % update K'
        Kp(s_ind,:,i) = Kp(s_ind, :, i) + lambda * Kp(s_ind-1,:,i);
        
        % loop over second string length
        for t_ind = 2:len2+1
            
            % indices in s2 such that s1(s_ind) matches entry of s2
            s2_inds = find(s1(s_ind-1) == s2(1:t_ind-1));
<<<<<<< HEAD
            
            % sum over prev K' to get new K'
            Kp(s_ind, t_ind, i) = Kp(s_ind, t_ind, i) + ...
                sum(Kp(s_ind-1, s2_inds, i-1) * ...
                lambda.^(t_ind-1 - s2_inds + 2)');
=======

            for j = s2_inds
               Kp(s_ind, t_ind, i) = Kp(s_ind, t_ind, i) + Kp(s_ind-1, j, i-1) * lambda^(t_ind-1 - j + 2);
            end            
>>>>>>> origin/master
            
        end
    end
end

% calculate K
K = zeros(len1+1, 1);

% loop over first string length
for s_ind = 2:len1+1
<<<<<<< HEAD
    
    % update K
    K(s_ind) = K(s_ind-1) + sum(Kp(s_ind-1, (s1(s_ind-1) == s2), end-1), 2)...
    * lambda^2;
    
=======

%     fprintf('finding %s in %s\n', s1(s_ind-1), s2);
%     s2_inds = find(s1(s_ind-1) == s2);
    
%     K(s_ind) = K(s_ind-1);

%     for j = s2_inds
%     
%         K(s_ind) = K(s_ind) + Kp(s_ind-1, j, end-1) * lambda^2; 
% 
%     end

    K(s_ind) = K(s_ind-1) + sum(Kp(s_ind-1, (s1(s_ind-1) == s2), end-1), 2) * lambda^2;

>>>>>>> origin/master
end

% Return answer
k = K(end);

end







