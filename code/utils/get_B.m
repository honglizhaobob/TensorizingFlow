function [B,c] = get_B(C,i,u,cprev)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Hongli Zhao, Michael Lindsey

% Inputs
% i is current index for conditional sampling
% u is the vector (phi_i(x)), where x is (i-1)th sampled variable (not needed if (i==1)
% cprev is the transformed preceding core (not needed if i==1), 
% should be a matrix size of size n by r, where r is the rank of the bond (i-1,i)
%
%
% Outputs
% returns matrix B, of size (legendre_ord x legendre_ord)
% returns new transformed core c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% external bond dimension (uniform for simplicity)
n = size(C{1},2);    
if i==1
    r = size(C{1},3);
    c = reshape(C{1},n,r);
    B = c*c';
else
    v = cprev'*u;
    c = einsum(C{i},v,1,1);
    B = c*c';
end