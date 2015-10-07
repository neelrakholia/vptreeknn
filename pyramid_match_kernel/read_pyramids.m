function [ dataset, indices ] = read_pyramids( filename )
% Reads in the pyramid structures from a .mrh file
% Stores a cell array of vectors of info

fid = fopen(filename, 'r');
    
num_points = fread(fid, 1, 'int32');
    
dataset = cell(num_points, 1);
indices = cell(num_points,0);

for i = 1:num_points
   
    num_bins = fread(fid, 1, 'int32');
    
    dataset{i} = zeros(2*num_bins, 1);

    for j = 1:num_bins
      
        index_size = fread(fid, 1, 'int32');
       
        if index_size > 0
            indices{i,j} = fread(fid, index_size, 'int32');
        end

        dataset{i}(2*(j-1)+1:2*j) = fread(fid, 2, 'double');

    end

    
end
    
fclose(fid);

% dataset = fread(fid, inf, 'int32')



end



