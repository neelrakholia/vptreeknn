

s1 = 'science is organized knowledge';
s2 = 'wisdom is organized life';

lambda = 0.5;

for n = 1:6
    
    tic
    k_ssk = ssk_fast(s1, s2, n, lambda);
    toc
    fprintf('ssk_fast: %g\n', k_ssk);
    
    
    tic
    k_dyn = ssk_dyn(s1, s2, n, lambda);
    toc
    fprintf('ssk_dyn: %g\n', k_dyn);
    
end


