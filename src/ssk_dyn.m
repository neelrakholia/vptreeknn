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
            s2_inds = find(strcmp(s2(1:t_ind-1),s1(s_ind-1)));
            
            % sum over prev K' to get new K'
            Kp(s_ind, t_ind, i) = Kp(s_ind, t_ind, i) + ...
                sum(Kp(s_ind-1, s2_inds, i-1) * ...
                power(lambda,(t_ind-1 - s2_inds + 2))');
            
        end
    end
end

% calculate K
K = zeros(len1+1, 1);

% loop over first string length
for s_ind = 2:len1+1
    
    % update K
    K(s_ind) = K(s_ind-1) + sum(Kp(s_ind-1, ...
        strcmp(s2(1:t_ind-1),s1(s_ind-1)), end-1), 2)...
    * lambda^2;
end

% Return answer
k = K(end);

end