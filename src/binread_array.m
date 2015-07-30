function data = binread_array(filename,numel)
%--------------------------------------------------------------------------
% BINREAD_ARRAY Reads contents of a binary file to an array
%   Input 
%       filename - file where contents are stored
%       numel - number of elements in the file
%
%   Output 
%       data - array of contents
%--------------------------------------------------------------------------

if nargin<1, check(); data=[]; return; end;
if nargin<2, numel=inf; end;

endiancheck_string = '1234ABCD';
endiancheck = hex2dec(endiancheck_string);

fid = fopen(filename,'r');
edc = fread(fid,1,'int64');
%assert(edc == endiancheck);
data = fread(fid,numel,'double');
fclose(fid);

%----
function check()
n = 3;
d = 2;
mat = randn(n, d);
binwrite_array(mat,'mat.bin');

matck=binread_array('mat.bin');
matck=reshape(matck,n,d);

assert(norm(matck-mat)==0)

disp('A-ok!');



