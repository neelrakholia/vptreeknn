% script to test ssk_dyn vs ssk_fast

% parameters
s1 = 'science is organized knowledge';
s2 = 'wisdom is organized life';

lambda = 0.5;

% for various subsequence lengths
for n = 1:6
    
    % ssk_fast time
    tic
    k_ssk = ssk_fast(s1, s2, n, lambda);
    toc
    fprintf('ssk_fast: %g\n', k_ssk);
    
    % ssk_dyn time
    tic
    k_dyn = ssk_dyn(C{2}, C{3}, n, lambda);
    toc
    fprintf('ssk_dyn: %g\n', k_dyn);
    
end


