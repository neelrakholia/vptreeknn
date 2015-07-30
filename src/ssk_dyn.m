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
K = zeros(len1 + 1, len2 + 1, sublen + 1);
Kp = zeros(len1 + 1, len2 + 1, sublen + 1);
Kpp = zeros(len1 + 1, len2 + 1, sublen + 1);

% consider the base cases
K(:,:,1) = 0;
Kp(:,:,1) = 1;

% case when min(|s|,|t|) < i
[x,y,z] = meshgrid(1:len2 + 1, 1:len2 + 1, sublen + 1);
K((z - 1) > x) = 0;
K((z - 1) > y) = 0;
Kp((z - 1) > x) = 0;
Kp((z - 1) > y) = 0;

% update K'
for i = 1:len1
    Kp(i + 1,:,:) = lambda*Kp(i,:,:) 

end

