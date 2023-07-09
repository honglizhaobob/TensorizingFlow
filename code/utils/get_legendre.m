function y = get_legendre(x,p,on_flag)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author(s): Michael Lindsey
%
% Custom routine for generating Legendre polynomials.
% 
% on_flag = true if orthonormal, false if monic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = length(x);
x = reshape(x,m,1);

y = zeros(m,p+1);
y(:,1) = 1;

if p<=1
    return;
end

y(:,2) = x;

for n=1:p-1
    k=n+1;
    y(:,k+1) = (1/(n+1))*( (2*n+1)*x.*y(:,k) - n*y(:,k-1) );
end

if ~on_flag
    return
end

for n=0:p
    k=n+1;
    y(:,k) = y(:,k)/sqrt(2/(2*n+1));
end