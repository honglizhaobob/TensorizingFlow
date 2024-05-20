%% My func 1
function [B] = TRtrial(Tin,B)
%Compress Tin (cell(1,n)) using guess
%to a smaller tensor using ALS
%need to define n: node #, r: tensor rank
%Tin: heavy tensor, B: initial guess for compression
tolerance = 10 ^(-8);
n = numel(Tin);
self = cell(1,n); cross = self; const = self;
for i=1:n
    cross{i} = permute(tns_mult(B{i},2,Tin{i},2),[1 3 2 4]);
    self{i}  = permute(tns_mult(B{i},2,B{i},2),[1 3 2 4]);
    
    const{i} = permute(tns_mult(Tin{i},2,Tin{i},2),[1 3 2 4]);
    
end


%Start ALS
%niter = 50*n;
for ii=1
    if rem(ii-1,n)==0
        idx_order = randperm(n);
    end
    
    i = mod(ii-1,n)+1;
    
    shift = circshift((1:n)',-(idx_order(i)-1));
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
    
    %Construct coefficient
    N = self{2};
    M = cross{2};
    for j=3:n
        N = tns_mult(N,[3,4],self{j},[1 2]);
        M = tns_mult(M,[3,4],cross{j},[1 2]);
    end
    
    N_size = size(N);
    N = reshape(permute(N,[3 1 4 2]),N_size(3)*N_size(1),N_size(4)*N_size(2));
    
    M = tns_mult(Tin{1},[3 1],M,[2,4]);
    M = permute(M,[1 3 2]);
    M_size = size(M);
    M = reshape(M,M_size(1),prod(M_size(2:3)));
    
    %Construct coefficient2
    N_2 = TRproduct(self(2:n),[3,4],[1,2]);
    M_2 = TRproduct(cross(2:n),[3,4],[1,2]);
    
    N_size_2 = size(N_2);
    N_2 = reshape(permute(N_2,[3 1 4 2]),N_size_2(3)*N_size_2(1),N_size_2(4)*N_size_2(2));
    
    M_2 = tns_mult(Tin{1},[3 1],M_2,[2,4]);
    M_2 = permute(M_2,[1 3 2]);
    M_size_2 = size(M_2);
    M_2 = reshape(M_2,M_size_2(1),prod(M_size_2(2:3)));
    
    %Difference in product:
    n_N = norm(N-N_2,'fro')
    m_M = norm(M-M_2,'fro')
    
end
end
