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

% construct 3, 3-dimensional arrays to store K, K', and K''
len1 = length(s1);
len2 = length(s2);

% initialize arrays
% first string, second string, subsequence (n)
% K = zeros(len1 + 1, len2 + 1, sublen + 1);
Kp = zeros(len1 + 1, len2 + 1, sublen + 1);
% Kpp = zeros(len1 + 1, len2 + 1, sublen + 1);

% consider the base cases
% K(:,:,1) = 0;
% Kp(:,:,1) = 1;

% case when min(|s|,|t|) < i
% [x,y,z] = meshgrid(1:len1 + 1, 1:len2 + 1, sublen + 1);
% K((z - 1) > x) = 0;
% K((z - 1) > y) = 0;
% Kp((z - 1) > x) = 0;
% Kp((z - 1) > y) = 0;
% 
% % update K'
% for i = 1:len1
%     
%     for j = 1:len2
%        
%         for l = 1:sublen
%            
%             
%             
%         end
%         
%     end
%     
%     Kp(i + 1,:,:) = lambda*Kp(i,:,:) + Kpp(i+1, :, :);
% 
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Kp(:,1,:) = 0;
Kp(1,:,:) = 0;
Kp(:,:,1) = 1;

for i = 2:sublen+1
   
    for s_ind = 2:len1+1
    
        Kp(s_ind,:,i) = Kp(s_ind, :, i) + lambda * Kp(s_ind-1,:,i);

        for t_ind = 2:len2+1

            % indices in s2 such that s1(s_ind) matches entry of s2
            s2_inds = find(s1(s_ind-1) == s2(1:t_ind-1));

            for j = s2_inds
               Kp(s_ind, t_ind, i) = Kp(s_ind, t_ind, i) + Kp(s_ind-1, j, i-1) * lambda^(t_ind-1 - j + 2);
            end
            
        end
        
    end
    
end


K = zeros(len1+1, 1);

for s_ind = 2:len1+1

%     fprintf('finding %s in %s\n', s1(s_ind-1), s2);
    s2_inds = find(s1(s_ind-1) == s2);
    
    K(s_ind) = K(s_ind-1);

    for j = s2_inds
    
        K(s_ind) = K(s_ind) + Kp(s_ind-1, j, end-1) * lambda^2; 

    end
    
end

k = K(end);















