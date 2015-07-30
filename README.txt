Tests nearest neighbor search for arbitrary distance metrics using vp-trees
--tree-type: vantange point trees with random selection of vantage points
--search: construct a number of vp-tree and iteratively search them for NN

----------------------------- FILES ------------------------------

% Test Drivers

Directory: ./test/

Files:
1) susy_rand.m                  Test for SUSY dataset
2) gaussian_rand.m              Test for gaussian distribution of points
3) covtype_rand.m               Test for Cover Type dataset


% Tree construction 

Directory: ./src/

Files:
1) classify_vp.m                Makes a single split of data points into 
                                left and right nodes
2) bsttree_vp.m                 Recursively constructs a binary search tree.
                                Also implements search methods decribed above.


% Distance and kernel evaluations

Directory: ./src/

Files:
1) ssk_fast.m                   Recursive implementation of ssk computation 
                                from an online source
2) ssk.m                        Recursive implementation of ssk computation
                                using my implementation
3) rbf.m                        Compute rbf kernel distance 
4) kknn.m                       Computes NN using the brute force quadratic 
                                search algorithm
5) distk.m                      Calls the appropriate kernel distance 
                                function


% Helper files 

Directory: ./src/

Files:
1) readfiles.m                  Read documents from a folder to get an array 
                                of strings
2) randomvp.m                   Used to run nn search for test drivers
3) generate_points.m            Generate points for gaussian_rand.m
4) binwrite_array.m             Write array contents to a binary files
5) binread_array.m              Read binary files to an array


% Files no used

Directory: ./src/obsolete/

These files use an O(n^2) tree construction and are not used anymore
