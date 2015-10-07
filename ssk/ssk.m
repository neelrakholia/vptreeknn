function value = ssk(s1, s2, sublen, lambda)
%--------------------------------------------------------------------------
% SSK Computes the vakue of string subsequence kernel using recursion
%   Input
%       s1 - string A
%       s2 - string B
%       sublen - length of subsequence
%       lambda - tuning parameter
%   Ouput
%       value - evaluated kernel value
%--------------------------------------------------------------------------
% base case
if(isempty(s1) == 1 || isempty(s2) == 1)
    value = 0;
    return;
end

% base case
if(min(length(s1),length(s2) < sublen))
    value = 0;
    return;
end

% get the sum
sum = 0;
j = find(s2 == s1(end));
for i = j
   sum = sum + kprime(s1(1:end-1),s2(1:i-1),sublen - 1,lambda)*lambda^2;
end
    
% recursive call 
value = ssk(s1(1:end-1),s2,sublen,lambda) + sum;
end


function value = kprime(s1, s2, sublen, lambda)
%--------------------------------------------------------------------------
% KPRIME auxillary function for ssk and kprimeprime
%   s1 - string A
%   s2 - string B
%   sublen - length of subsequence
%   lambda - tuning parameter
%--------------------------------------------------------------------------
% base cases
if(sublen == 0)
    value = 1;
    return;
end

if(isempty(s1) == 1 || isempty(s2) == 1)
    value = 0;
    return;
end

if(min(length(s1),length(s2) < sublen))
    value = 0;
    return;
end

% recursive call
value = kprimeprime(s1,s2,sublen,lambda) + ...
lambda*kprime(s1(1:end-1),s2,sublen,lambda);

end


function value = kprimeprime(s1,s2,sublen,lambda)
%--------------------------------------------------------------------------
% KPRIMEPRIME auxillary function for ssk and kprimeprime
%   s1 - string A
%   s2 - string B
%   sublen - length of subsequence
%   lambda - tuning parameter
%--------------------------------------------------------------------------
% base case
if(isempty(s1) == 1 || isempty(s2) == 1)
    value = 0;
    return;
end

% recursive call as in the paper
value = lambda*(lambda*kprime(s1(1:end-1),s2(1:end-1),sublen-1,lambda) + ...
kprimeprime(s1,s2(1:end-1),sublen,lambda));

end