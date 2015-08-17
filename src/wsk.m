function d = wsk(X, x, sigma, sublen)
%--------------------------------------------------------------------------
% WSK evaluates the word sequence kernel distance between pairs of points 
% in X and x
%   Input 
%       X - Array of points
%       x - Array of points
%       sigma - Lambda value desired
%       sublen - word sequence length
%
%   Output 
%       d - evaluated kernel value
%--------------------------------------------------------------------------

% get lengths of arrays
n = length(x);
N = length(X);

% initialize output array
d = zeros(N,n);
X_self = zeros(N,1);
x_self = zeros(n,1);

% pre-compute self kernel values
for k = 1:n
    string1 = strjoin(x{k},',');
    x_self(k) = ssk_dyn_mex(string1,length(x{k}),string1,length(x{k}),...
        sublen,sigma,length(string1),length(string1));
end

for k = 1:N
    string1 = strjoin(X{k},',');
    X_self(k) = ssk_dyn_mex(string1,length(X{k}),string1,length(X{k}),...
        sublen,sigma,length(string1),length(string1));
end

% loop over the matrix and compute lengths
for i = 1:N
    for j = 1:n
        string1 = strjoin(X{i},',');
        string2 = strjoin(x{j},',');
        d(i,j) = - 2 * ssk_dyn_mex(string1,length(X{i}),string2,...
            length(x{j}),sublen,sigma,length(string1),length(string2))/...
            sqrt(X_self(i)*x_self(j));
    end
end