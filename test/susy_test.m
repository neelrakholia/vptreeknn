% read train
filename = 'susy.icml.trn.X.bin';
n = 4499999;
dim = 18;
train = binread_array(filename, n*dim);
train = reshape(train, dim, n);

% read test
filename = 'susy.icml.tst.X.bin';
m = 499999;
test = binread_array(filename, m*dim);
test = reshape(test, dim, m);

% sample data
n = 100000;
m = 1000;
train = datasample(train, n, 2, 'Replace', false);
test = datasample(test, m, 2, 'Replace', false);

% sample data to form kernel matrix
p = 300;
rp = randperm(n);
ptrain = train(:,rp(1:p));
query = train(:,rp);

% get other data points
pother = train(:,rp(p + 1:end));

% for kernel matrices
Ktrain = real(distk(ptrain,ptrain,0.22));
Ktest = real(distk(test,ptrain,0.22));
Kother = real(distk(pother,ptrain,0.22));

%form KLSH hash table
[H W] = createHashTable(Ktrain,p,30);
% hash query points and other data points
H_other = (Kother*W)>0;
H_query = (Ktest*W)>0;
H_all = vertcat(H, H_other);

%form distance matrix between queries and database items
CH = compactbit(H);
CH_query = compactbit(H_query);
CH_other = compactbit(H_other);
CH_all = vertcat(CH, CH_other);

epsilon = 0.5;
M = floor(n^(1/(1+epsilon)));
candidate_nn = zeros(m,2*M);

% CH has the hashed values of the database points.
for i = 1:M
    i
    % a random permutation of the b bits
    perm = randperm(p);
    % combine all values together
    H_dq = vertcat(H_all, H_query);
    % permute all the data base entries, then sort
    [sorted_vals,sorted_inds] = sortrows((H_dq(:,perm)));
    % get indices for queries
    q_ind = (n+1):(n+m);
    % find queries
    [~,qin_copy] = ismember(q_ind,sorted_inds);
    % find candidates
    indi = 1:numel(sorted_inds);
    qin = qin_copy;
    left = m;
    
    % for left neighbors
    j = 1;
    while(left ~= 0 && j < m/4)
        % get neighboring points
        q_left = qin - j;
        
        % check for its validity
        q_left_valid = find(q_left > 0);
        q_left_inds = find(sorted_inds(q_left(q_left_valid)) <= n);
        
        % get all the valid points
        potential = sorted_inds(q_left(q_left_valid));
        left_candidates = potential(q_left_inds);
        
        % store valid candidates
        candidate_nn(q_left_valid(q_left_inds),2*i - 1) = left_candidates;
        
        % make all other values zero
        q_left(q_left_valid(q_left_inds)) = 0;
        qin = q_left;
        
        % determine remaining candidates to be assigned
        left = sum(qin > 0);
        j = j + 1;
    end
    
    % for right neighbors
    qin = qin_copy;
    left = m;
    j = 1;
    while(left ~= 0 && j < m/4)
        % get neighboring points
        q_right = qin + j;
        
        % check for its validity
        q_right_valid = find(q_right <= (n + m) & q_right > 0);
        q_right_inds = find(sorted_inds(q_right(q_right_valid)) <= n);
        
        % get all the valid points
        potential = sorted_inds(q_right(q_right_valid));
        right_candidates = potential(q_right_inds);
        
        % store valid candidates
        candidate_nn(q_right_valid(q_right_inds),2*i) = right_candidates;
        
        % make all other values zero
        q_right(q_right_valid(q_right_inds)) = -inf;
        qin = q_right;
        
        % determine remaining candidates to be assigned
        left = sum(qin > 0);
        j = j + 1;
    end

%     for j = 1:m
%         % the two candidate neighbors are the entries that come right
%         % before and after the permuted query
%         potential_candi_greater = find(indi > qin(j), 10);
%         potential_candi_lesser = find(indi < qin(j), 10, 'last');
%         candidate1 = find(sorted_inds(potential_candi_greater) < n,1);
%         candidate2 = find(sorted_inds(potential_candi_lesser) < n,1,'last');
%         % if the query is at the end
%         if(isempty(candidate1))
%             candidate2 = find(sorted_inds(potential_candi_lesser) < n,2,'last');
%             candidates = sorted_inds(potential_candi_lesser(candidate2));
%         else
%             % if query is in the beginning
%             if(isempty(candidate2))
%                 candidate1 = find(sorted_inds(potential_candi_greater) < n,2);
%                 candidates = sorted_inds(potential_candi_greater(candidate1));
%             else
%                 candidates = [sorted_inds(potential_candi_lesser(candidate2)),...
%                     sorted_inds(potential_candi_greater(candidate1))];
%             end
%         end
%         candidate_nn(j,(2*i - 1):2*i) = candidates;
        % get nearest neighbor candidates
%         if(qin < (n+1) && qin > 1)
%             candidate_nn(j,(2*i - 1):2*i) = [sorted_inds(qin - 1); ...
%                 sorted_inds(qin + 1)]';
%         else
%             if(qin > 1)
%                 candidate_nn(j,(2*i - 1):2*i) = [sorted_inds(qin - 2); ...
%                     sorted_inds(qin - 1)]';
%             else
%                 candidate_nn(j,(2*i - 1):2*i) = [sorted_inds(qin + 2); ...
%                     sorted_inds(qin + 1)]';
%             end
%         end
    %end
    
end


Dist = hammingDist(CH_query,CH);

% form distance matrix between other points and query points
Dist_other = hammingDist(CH_query,CH_other);

% combine distance matrices
dist_all = horzcat(Dist,Dist_other);

% define variables
in = 1:n;
k = 5;

% compute actual nn
piece = 100;
actual_nn = zeros(m,k);
actual_nn(1:m/piece, :) = kknn(query,in,test(:,1:m/piece),0.22,k,n);
for i = 2:piece
    actual_nn((i - 1)*(m/piece) + 1:i*m/piece,:) = ...
        kknn(query,in,test(:,(i - 1)*(m/piece) + 1:i*m/piece),0.22,k,n);
end
%actual_nn = kknn(query, in, test, 0.22, k, n);
points_nn = zeros(m,k);

% points = reshape(candidate_nn, [1,m*M*2]);
% candidates = unique(points);
% points_nn = kknn(query,candidates,test,0.22,k,numel(candidates));

% replace all zeros with 1
candidate_nn(candidate_nn == 0) = 1;

% get nearest neighbors
len = 0;
for i = 1:m
    candidates = unique(candidate_nn(i,:));
    points_nn(i,:) = kknn(query,candidates,test(:,i),0.22,k,...
        numel(candidates));
    len = len + numel(candidates);
    %points_nn(i,:) = candidates(ids);
    %[v, ind] = sort(dist,'ascend');
    %points_nn(i,:) = candidate_nn(i,ind(1:k));
end

% calculate accuracy
sum = 0;
for i = 1:m
    sum = sum + length(intersect(actual_nn(i,:),points_nn(i,:)));
end
acc = sum/(m*k);

disteval = (p*p + len)/(n*m)