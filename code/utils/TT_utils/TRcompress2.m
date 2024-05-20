function [B,ERR] = TRcompress2(Tin,B)
%Compress Tin (cell(1,n)) using guess
%to a smaller tensor using ALS
%need to define n: node #, r: tensor rank
%Tin: heavy tensor, B: initial guess for compression
tolerance = 10 ^(-4);
n = numel(Tin);
self = cell(1,n); cross = self; const = self;
for i=1:n
    cross{i} = permute(tns_mult(B{i},2,Tin{i},2),[1 3 2 4]);
    self{i}  = permute(tns_mult(B{i},2,B{i},2),[1 3 2 4]);
    
    const{i} = permute(tns_mult(Tin{i},2,Tin{i},2),[1 3 2 4]);
    
end

%Tensor ring norm
P_sqrt = TRnorm(Tin);
P = P_sqrt.^2;

%Start ALS
%niter = 50*n;
niter = 500;
ERR(1) = TRnorm_diff(Tin,B)/P_sqrt;
for ii=1:niter
    if rem(ii-1,n)==0
        idx_order = randperm(n);
    end
    
    i = mod(ii-1,n)+1;
    
    shift = circshift((1:n)',-(idx_order(i)-1));
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
    
    %Construct coefficient
%     N = self{2};
%     M = cross{2};
%     for j=3:n
%         N = tns_mult(N,[3,4],self{j},[1 2]);
%         M = tns_mult(M,[3,4],cross{j},[1 2]);
%     end
    N = TRproduct(self(2:n),[3,4],[1,2]);
    M = TRproduct(cross(2:n),[3,4],[1,2]);
    
    N_size = size(N);
    N = reshape(permute(N,[3 1 4 2]),N_size(3)*N_size(1),N_size(4)*N_size(2));
    
    M = tns_mult(Tin{1},[3 1],M,[2,4]);
    M = permute(M,[1 3 2]);
    M_size = size(M);
    M = reshape(M,M_size(1),prod(M_size(2:3)));
    
    Bcurr = B{1};
    alpha = 10^-7;
    B_size = size(B{1});
    tmp   = (M + alpha * reshape(permute(Bcurr,[2 1 3]),M_size(1),prod(M_size(2:3))))/(N + alpha * eye(size(N)));
    err   = sqrt(abs(trace(tmp*N*tmp')-2*trace(M*tmp')+P)/P);
    %ERR(ii+1) = err;
    B{1} = permute(reshape(tmp,B_size(2),B_size(1),B_size(3)),[2 1 3]);
    %norm(B{1}(:)-Bcurr(:));
    self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
    cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
    
    shift = circshift((1:n)',idx_order(i)-1);
    
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
    %display(['Iter = ' num2str(ii) ' Error = ' num2str(err) ]);
    
    ERR(ii+1) = TRnorm_diff(Tin,B)/P_sqrt;
    
    %if (err < tolerance)
    if (ERR(end) < tolerance)
        break;
    end
    
end
%display(['Iter = ' num2str(numel(ERR)) ' Error = ' num2str(err) ]);
%display('Last error')
%TRnorm(TRsubtract(Tin,B))
end
