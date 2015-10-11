function cb = compactbit(b)
% IMPORTANT: number of bits needs to be small enough to fit in a uint64
% more than that, it has to be few enough bits to fit in a double 
% correctly


% Idea is now to always put the whole row in a single MATLAB word
    
[nSamples, nbits] = size(b);

% % % powers of 2
% powers = 2.^(0:nbits-1);
%  
% % % now it's just a matvec
% cb = uint64(b * powers');

% cb = num2str(b);

% cb = char(b + '0');




% 
% 
nwords = ceil(nbits/8);
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end
