function [ val ] = bitlt( x, y )


% compare vectors x and y lexicographically from left to right
% 
% ltres = x < y;
% gtres = x > y;
% val = ltres(xor(ltrres, gtres));
% val = val(1); 


ltres = bsxfun(@lt, x, y);
gtres = bsxfun(@gt, x, y);
[~,inds] = max(xor(ltres,gtres), [], 2);

val = ltres(sub2ind(size(ltres), [1:length(inds)]',inds));

end

