function xx = get_sample(C,A,xg)
% Authors: Hongli Zhao, Michael Lindsey
%
% Part of the conditional sampling routine. Computes a sample 
% coordinate.
%
% C is the coefficient train which is a compact
% representation of the functional series
%
% A should contain all Legendre data matrices
% for each dimension (with appropriate range)
%
% likewise, xg should contain all query grid points
% of [-1, 1] for each dimension.


d = length(size(C));
assert(length(A)==d, "> need to have enough Legendre data.")
assert(length(xg)==d, "> double check multidimensional grid. ")
% number of Legendre polynomials used
n = size(A{1},1);
p = n-1;

% get first B
[B,c] = get_B(C,1);
xx = zeros([1,d]);
for i=2:d
    % get dimension (i-1) Legendre data
    A_i = A{i-1};
    xg_i = xg{i-1};    
    % form prob vector from preceding B
    pp = sum(A_i.*(B*A_i),1);
    % prevent numerical zeros 
    pp = max(pp,0);
    pp = pp/sum(pp);
    % sample x from vector and perturb
    x = randsample(xg_i,1,true,pp);
    % uniform perturbation should be scaled accordingly

    % compute all spacings (not uniform for legendre)
    spacings = xg_i(2:end)-xg_i(1:end-1);
    % find which point on the legendre grid is sampled
    legendre_idx = find(x == xg_i);
    % to prevent error, perturb the last point the same way
    % as (N-1)th point
    if legendre_idx == length(xg_i)
        legendre_idx = legendre_idx - 1;
    end
    % perturb with appropriately scaled uniform noise
    dx_i = spacings(legendre_idx);
    x = x+((dx_i*randn)-(dx_i/2));
    
    % store this sample
    xx(i-1) = x;
    % form vector of basis values at x
    u = get_legendre(x,p,true)';
    % get next B
    cprev = c;
    [B,c] = get_B(C,i,u,cprev);
end

% get last sample
A_i = A{end};
xg_i = xg{end};
pp = sum(A_i.*(B*A_i),1);
pp = max(pp,0);
pp = pp/sum(pp);

x = randsample(xg_i,1,true,pp);
% compute all spacings (not uniform for legendre)
spacings = xg_i(2:end)-xg_i(1:end-1);
% find which point on the legendre grid is sampled
legendre_idx = find(x == xg_i);
% to prevent error, perturb the last point the same way
% as (N-1)th point
if legendre_idx == length(xg_i)
    legendre_idx = legendre_idx - 1;
end
% perturb with appropriately scaled uniform noise
dx_i = spacings(legendre_idx);
x = x+((dx_i*randn)-(dx_i/2));
xx(d) = x;
end