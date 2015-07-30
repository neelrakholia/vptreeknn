function fid = binwrite_array( data, filename)
%--------------------------------------------------------------------------
% BINWRITE_ARRAY Writes contents of an array to a binary file
%   Input 
%       data - array to write
%       file - file where it needs to be written
%
%   Output 
%       fid - file id where array contents were written
%--------------------------------------------------------------------------

endiancheck_string = '1234ABCD';
endiancheck = hex2dec(endiancheck_string);

fid = fopen( filename, 'w');
fwrite(fid, endiancheck, 'int64');
fwrite(fid, data(:), 'double');




