function r = generate_points(dim, n, method)
%--------------------------------------------------------------------------
% GENERATE_POINTS Generates points from specified distribution
%   Input 
%       dim - dimension of the data
%       n - total number of points
%       method - the type of distribution to be used
%
%   Output 
%       r - generated points
%--------------------------------------------------------------------------
if nargin<3, method = 'uniform'; end

% different points distribution generation
switch method
   case 'gaussian'
      r = randn(dim, n);
   
   case 'line'
      x = rand(dim,1);
      dir = rand(dim,1);
      scal = rand(n,1)*n;
      for i=1:dim, r(i,:) = x(i)+scal*dir(i); end

   case 'line_in_ones'
      r=generate_points(dim,n,'line');
      r=r+ones(dim,n);

    case '4d_intrinsic'
      if dim<=4, r = generate_points(dim,n,'gaussian'); end;
      r4= randn(4,n);
      r = 0.001*generate_points(dim,n,'gaussian');
      r(1:4,:)=r4;
      A=rand(dim,dim); [U,~,~]=svd(A);
      r = U*r;
      
   otherwise %uniform case
      r = rand( dim, n);
end

end
