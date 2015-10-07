function [result, K] = ssk_fast(s,t,p,lambda)
%--------------------------------------------------------------------------
%SSK_FAST
%        -Finds the string subsequence kernel count between strings s and t
%         by using a recursive programming implementation,
%         where the length of the subsequence is p.
%        -Is faster than ssk b/c this program only tries to fill in the last
%         index of the matrix, rather than the whole structure.
%
%        -Simply prompting the function will return the value K(s,t), however
%         using the function as [result,K] = K(s,t) will also return the matrix K.
%
%        -The following algorithm is used:
%         K[p](sa,t) = K[p](s,t) + [Summation of i from 1 to |t|] K'[p-1](s,t(1:i-1))*(lambda^2) [t(i) == a]
%           K[p](s,t) = 0 if |s| < p  or |t| < p
%         K'[p](sa, t) = lambda*K'[p](s,t) + [Summation of i from 1 to |t|] K'[p-1](s,t(1:i-1)*(lambda^(|t|-j+2)  [t(i) == a]
%           K'[0](s,t) = 1 for all s,t
%           K'[p](s,t) = 0 if |s| < p  or |t| < p
%         
%
%        -Example: ssk_fast('abccc','abc', 3, 1) returns a value of 3.
%            (Note that ssk_fast('abccc','abc',3, 1)=ssk_fast('abc','abccc',3, 1) since K(s,t) = K(t,s) ).
%        -Example: ssk_fast('abccc','abc', 3, 2) returns a value of 448.
%        -Example: ssk_fast('a','a', 1) returns a value of 1.
%        -Example: ssk_fast('a','b', 1) returns a value of 0.
%        -Example: ssk_fast('ab','ab', 2) returns a value of 1.
%         
%
%
%USAGE:   scalar = ssk_fast('string1','string2', p, lambda);
%                       (where p is the length of the substring and lambda is the cost of penalties)
%
%         [scalar, matrix] = ssk_fast('string1,'string2', p, lambda);
%
%

%For more information, visit http://www.kernel-methods.net/
%
%Written and tested in Matlab 6.0, Release 12.
%Copyright 2003, Manju M. Pai 4/2003
%manju@kernel-methods.net
%--------------------------------------------------------------------------

%Obtain lengths of strings
[num_rows_s, n] = size(s);
[num_rows_t, m] = size(t);

%Error checking statements:
  %If p is longer than either string, the answer is obviously zero
    if (p > n && p > m)
        result = 0;
        return
    end;

  %Make sure input vectors are horizontal.
  %if ~( xor( (num_rows_s < 2), (n < 2)) & xor( (num_rows_t < 2), (m < 2) ) )  
  %   error('Error: s and t must be vectors.');
  %end;
  
  %If p is less than one or not an integer, program should quit due to faulty variable input.
  if p <= 0 || ischar(p) || mod(p,1) ~= 0
      error('Error: p needs to be an integer greater than 0.');
  end;
  
  %If lambda is less than one or not a number, program should quit due to faulty variable input.
  if lambda <= 0 || ischar(lambda)
      error('Error: lambda needs to be a number greater than 0.');
  end; 
  
  %Turn vertical vectors into horizonal vectors
  if( n == 1 && num_rows_s > 1 )
      s = s';
      [~, n] = swap(num_rows_s, n);
  end;
  if (m == 1 && num_rows_t > 1 )
      t = t';
      [~, m] = swap(num_rows_t, m);
  end;
  
%End of error checking


%Initially set every matrix index to -1 to show value has not yet been found
K = repmat(-1, [n, m]);                %The main kernel
K_prime = repmat(-1, [n, m, p]);             %The suffix kernel


%Fill in the rest of the matrix using the function ssk_fast_kernel()
[K(n,m), ~] = ssk_fast_kernel(s, t, K, K_prime, p, lambda);

result = K(n,m);

%------------------------------------------------------------------------------------------

function [answer, K_prime] = ssk_fast_kernel(sa, t, K, K_prime, p, lambda)
%This function is called by ssk_fast().
%Type 'help ssk_fast' for a description of the program.
%

%------------------------------------------------------------------------------------------

%Obtain lengths of both strings
n = length(sa);
m = length(t);

%truncate last character of string and obtain length of new string
s = sa(1:n-1);
length_s = length(s);

%Start algorithm:
  % 1) Split main algorithm into two parts:
    % a) K[p](s,t)
       if (length(s) < p) || (length(t) < p)
         %This is a base case where 0 is returned if either string has length 0
         answer = 0;
       elseif( K( length(s), length(t) ) == -1 )
         % Value has not yet been calculated
         answer = ssk_fast_kernel(s, t, K, K_prime, p, lambda);
       else
         % Value has already been calculated
         answer = K( length(s), length(t) );
       end;

    % b) Summation of K_prime[p-1](s,t(1:i-1))[t(i) == a] for  i = 1:(length(t))
    
      %this is the letter (a) that was truncated off the string
      letter = sa(n);

      %We need this 'for' loop as a cursor that iterates through the t string.
      pos_array = find(t(1:(m)) == letter);  %array which consists of all indices of t where t(i) == a
      for index = 1:length(pos_array)
        i = pos_array(index);
        length_t = length(t(1:(i-1)));
        if ( (p-1) == 0 )
          result = 1;
        elseif (length_s < (p-1) || length_t < (p-1))
          %This is a base case where 0 is returned if either string has length 0
          result = 0;
        elseif ( K_prime( length_s, length_t, (p-1)) == -1 )
          % Value has not yet been calculated
          [result, K_prime] = suffix_kernel(s, t(1:(i-1)), K_prime, (p-1), lambda);
        else
          % Value has already been calculated
          result = K_prime( length_s, length_t, (p-1));
        end;
        answer = answer + (result*lambda*lambda);
      end;
            
      return
% End of algorithm



%------------------------------------------------------------------------------------------

function [answer, K_prime] = suffix_kernel(sa, t, K_prime, i, lambda)
%This function is called by ssk_fast().
%Type 'help ssk_fast' for a description of the program.
%

%------------------------------------------------------------------------------------------

%Obtain lengths of both strings
n = length(sa);
m = length(t);

%truncate last character of string and obtain length of new string
s = sa(1:n-1);
length_s = length(s);

%Start algorithm:
  % 1) Split main algorithm into two parts:
    % a) lambda * K_prime[i](s,t)
       if (length(s) < i) || (length(t) < i)
         %This is a base case where 0 is returned if either string has length 0
         answer = 0;
       elseif( K_prime( length(s), length(t) ) == -1 )
         % Value has not yet been calculated
         answer = lambda * suffix_kernel(s, t, K_prime, i, lambda);
       else
         % Value has already been calculated
         answer = lambda * K_prime( length(s), length(t) );
       end;

    % b) Summation of K_prime[p-1](s,t(1:j-1))[t(i) == a] for  j = 1:(length(t))
    
      %this is the letter (a) that was truncated off the string
      letter = sa(n);

      %We need this 'for' loop as a cursor that iterates through the t string.
      pos_array = find(t(1:(m)) == letter);  %array which consists of all indices of t where t(j) == a
      for index = 1:length(pos_array)
        j = pos_array(index);
        length_t = length(t(1:(j-1)));
        if ( (i-1) == 0 )
          result = 1;
        elseif (length_s < (i-1) || length_t < (i-1))
          %This is a base case where 0 is returned if either string has length 0
          result = 0;
        elseif ( K_prime( length_s, length_t, (i-1)) == -1 )
          % Value has not yet been calculated
          [result, K_prime] = suffix_kernel(s, t(1:(j-1)), K_prime, (i-1), lambda);
        else
          % Value has already been calculated
          result = K_prime( length_s, length_t, (i-1));
        end;
        answer = answer + (result*(lambda^(m-j+2)));
      end;
            
      return
% End of algorithm

%------------------------------------------------------------------------------------------

function [x, y] = swap(x, y)
%swaps values so that x = y and y = x

%------------------------------------------------------------------------------------------

temp = x;
x = y;
y = temp;

return;