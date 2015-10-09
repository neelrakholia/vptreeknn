function [ kval ] = cosine_kernel( x, y )

normx = sqrt(sum(x.^2, 1));
normy = sqrt(sum(y.^2, 1));

kval = x'*y ./ (normx' * normy);

end

