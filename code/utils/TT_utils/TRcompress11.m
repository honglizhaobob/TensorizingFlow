function [B,ERR] = TRcompress11(Tin,B,r,total_rounds)
%Compress Tin (cell(1,n) to a smaller tensor using ALS
%need to define n: node #, r: tensor rank
%Tin: heavy tensor, B: initial guess for compression
tol = 10 ^(-4);
n = numel(Tin);
 alpha = 10^-7;
 tau = 10^(-2);
Tin_size = size(Tin{1});
actual_r = Tin_size(1);
min_iter_inner = 20;
min_iter_outer = 5;
min_iter = min_iter_outer * n;
max_correct = 10;

%min_iter = (100 * (actual_r - r) + 3);

%B = Tin;

%B = TRinitial(Tin,actual_r);

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
%niter = 1000*n;

ERR = 0;
index_e =1;

for ii=1:(total_rounds*n)
    %Sequential ordering
    if rem(ii-1,n)==0
        %if ii<2
            idx_order = 1:n;
        %else
           %idx_order = randperm(n);
        %end
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
    
    %Initialize with SVD
    %if (mod(ii-1,n)<1
%        B_12 = tns_mult(B{n},3,B{1},1);
%        B_12_size = size(B_12);
%        B_12 = reshape(B_12,prod(B_12_size(1:2)),prod(B_12_size(3:4)));
%        [U,S,V] = svd(B_12,'econ');
%        S = sqrt(S((1:actual_r),(1:actual_r)));
%        U = U(:,1:actual_r)*S;
%        V = S*V(:,1:actual_r)';
% 
% 
%        B{n} = reshape(U,B_12_size(1),B_12_size(2),actual_r);
%        B{1} = reshape(V,actual_r,B_12_size(3),B_12_size(4));
%        self{n}  = permute(tns_mult(B{n},2,B{n},2),[1 3 2 4]);
%        cross{n} = permute(tns_mult(B{n},2,Tin{n},2),[1 3 2 4]);
%        self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
%        cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
    %end
    
     %Set the outer matrix
    N_out = TRproduct(self,[3,4],[1,2]);
    M_out = TRproduct(cross,[3,4],[1,2]);
    
    %R_1=eye(actual_r,r)+alpha*randn(actual_r,r);
    R_1 = randn(size(B{n},3),r);
    %R_1=eye(actual_r,r);
    %R_2=eye(r,actual_r)+alpha*randn(r,actual_r);
    R_2 = randn(r,size(B{1},1));
    %R_2_st=eye(r,actual_r)+alpha*randn(r,actual_r);
    %R_2=eye(r,actual_r);
    %R_2(:,(r+1):(2*r))=eye(r)+alpha*rand(r);
    prev_err_1=10;
     prev_err_2=10;
     err_1=0;
      err_2=10;
    
    correct = 0;
      
   for j = 1:min_iter_inner
       
       temp_B = B; 
       temp_B{n} = tns_mult(temp_B{n},3,R_1,1);
       temp_B{1} = tns_mult(R_2,2,temp_B{1},1);
       
       real_error =TRnorm_diff(Tin,temp_B)/P_sqrt;
       ERR(index_e) = real_error;
       index_e = index_e+1;
       
       if (real_error < tol)||(abs(prev_err_1-err_1)<(10^(-8)) && abs(prev_err_2-err_2)<(10^(-8)))
           break
       end
       %Optimize for the first one:
       N = tns_mult(R_2,2, N_out,1);
       N = permute(tns_mult(R_2,2, N,2),[2,1,3,4]);
       M = tns_mult(R_2,2, M_out,1);
       
       N_size = size(N);
       N = reshape(permute(N,[3 1 4 2]),N_size(3)*N_size(1),N_size(4)*N_size(2));
       
       M = TRtrace(M,2,4,[1,3]);
       M_size = size(M);
       M = reshape(M',numel(M),1);
    
       
       Bcurr = reshape(R_1',1,numel(R_1));
       tmp   = pinv(N)*(M);
       %tmp = (N+alpha * eye(size(N)))\M;
       prev_err_1 = (R_1(:)'*N*R_1(:) - 2*M'*R_1(:)+P)/P;
       err_1   = (tmp'*N*tmp-2*M'*tmp+P)/P;
       
       if (abs(prev_err_1-err_1)<(10^(-8)))
          if (correct < max_correct)
           tmp = tmp + tau * (N*tmp - M+ alpha * tmp + norm(tmp)*randn(numel(tmp),1));
           correct = correct +1;
          else
              break;
          end
      end
       
       R_1 = reshape(tmp,M_size(2),M_size(1));
       [U,~,V] = svd(R_1);
      
       R_1 = U(:,1:r)*V(:,1:r)';
 
     %  max_R_1 = max(abs(R_1(:)));
     % max_R_2 = max(abs(R_2(:)));
     % delta_factor = sqrt(max_R_1 *max_R_2);
       
     %  R_1 = (delta_factor ./ max_R_1) .* R_1;
     %  R_2 = (delta_factor ./ max_R_2) .* R_2;
       
   
       
       temp_B = B; 
       temp_B{n} = tns_mult(temp_B{n},3,R_1,1);
       temp_B{1} = tns_mult(R_2,2,temp_B{1},1);
       
       real_error =TRnorm_diff(Tin,temp_B)/P_sqrt;
       ERR(index_e) = real_error;
       index_e = index_e+1;
       if real_error < tol
           break
       end
%        B{1} = permute(reshape(tmp,B_size(2),B_size(1),B_size(3)),[2 1 3]);
%        self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
%        cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
%        
       
       %Optimize for the secod one:
       N = permute(tns_mult(N_out,3,R_1,1),[1,2,4,3]);
       N = tns_mult(N,4,R_1,1);
       M = permute(tns_mult(M_out,3,R_1,1),[1,2,4,3]);
  
       N_size = size(N);
       N = reshape(permute(N,[3 1 4 2]),N_size(3)*N_size(1),N_size(4)*N_size(2));
       
       M = TRtrace(M,2,4,[1,3]);
       M_size = size(M);
       M = reshape(M',numel(M),1);
    
       Bcurr = reshape(R_2',1,numel(R_2));
       tmp   = pinv(N)*(M);
       %tmp = (N+alpha * eye(size(N)))\M;
       prev_err_2 = (R_2(:)'*N*R_2(:) - 2*M'*R_2(:)+P)/P;
       err_2   = (tmp'*N*tmp-2*M'*tmp+P)/P;
       
      if (abs(prev_err_2-err_2)<(10^(-8))) 
          if (correct < max_correct)
           tmp = tmp + tau * (N*tmp - M+ alpha * tmp + norm(tmp)*randn(numel(tmp),1));
           correct = correct+1;
          else
              break;
          end
      end
       
      R_2 = reshape(tmp,M_size(2),M_size(1));
%       max_R_1 = max(abs(R_1(:)));
%       max_R_2 = max(abs(R_2(:)));
%       delta_factor = sqrt(max_R_1 *max_R_2);
%        
%        R_1 = (delta_factor ./ max_R_1) .* R_1;
%        R_2 = (delta_factor ./ max_R_2) .* R_2;
       
      % R_2 = reshape(tmp,M_size(2),M_size(1));
       [U,~,V] = svd(R_2);
      
       R_2 = U(:,1:r)*V(:,1:r)';
       
%        B{1} = permute(reshape(tmp,B_size(2),B_size(1),B_size(3)),[2 1 3]);
%        self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
%        cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
%       
     
   end
   
    B{n} = tns_mult(B{n},3,R_1,1);
    self{n}  = permute(tns_mult(B{n},2,B{n},2),[1 3 2 4]);
    cross{n} = permute(tns_mult(B{n},2,Tin{n},2),[1 3 2 4]);
   
    B{1} = tns_mult(R_2,2,B{1},1);
    self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
    cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
      
    shift = circshift((1:n)',idx_order(i)-1);
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
    %display(['Iter = ' num2str(ii) ' Error = ' num2str(err) ]);
      
    
%     %if (ii >= min_iter) && (err < tolerance)
     if (ii >= n+1) && (ERR(end) < tol)
         break;
     end
    
end
end
