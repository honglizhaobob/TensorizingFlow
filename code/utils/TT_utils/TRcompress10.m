function [B,ERR] = TRcompress10(Tin,B,r,total_rounds)
%Compress Tin (cell(1,n) to a smaller tensor using ALS and gauges
%need to define n: node #, r: tensor rank
%Tin: heavy tensor, B: initial guess for compression
tol = 10 ^(-4);
n = numel(Tin);
alpha = 10^(-1);
tau = 10^(-2);
min_iter_inner = 30;
max_correct = 10;

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

ERR = 0;
index_e =1;

for ii=1:(total_rounds*n)
    %Sequential ordering
    if rem(ii-1,n)==0
            idx_order = 1:n;
    end
    
    i = mod(ii-1,n)+1;
   
    shift = circshift((1:n)',-(idx_order(i)-1)); 
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
        
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
    
        
    if ii < 2
    %R_1=repmat(eye(r),2,1);
     %R_2=R_1';
    else
        alpha = 10^(-8);
    end
    R_1 = randn(size(B{n},3),r);
    %R_1=eye(actual_r,r)+alpha*randn(actual_r,r);
    %R_1=repmat(eye(r),2,1);
    %R_1((r+1):actual_r,:)=eye(r);
    %R_2=eye(r,actual_r)+alpha*randn(r,actual_r);
    R_2 = randn(r,size(B{1},1));
    %R_2_st=eye(r,actual_r)+alpha*randn(r,actual_r);
    %R_2=R_1';
    %R_2(:,(r+1):(2*r))=eye(r)+alpha*rand(r);
    %end
   
    correct = 0;
   for j = 1:min_iter_inner
       
       %Compute the real error in the tensor
       temp_B = B; 
       temp_B{n} = tns_mult(temp_B{n},3,R_1,1);
       temp_B{1} = tns_mult(R_2,2,temp_B{1},1);
       
       real_error =TRnorm_diff(Tin,temp_B)/P_sqrt;
       ERR(index_e) = real_error;
       index_e = index_e+1;
       if (real_error < tol)
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
       
       
       tmp = (N+alpha * eye(size(N)))\M;
       prev_err_1 = (R_1(:)'*N*R_1(:) - 2*M'*R_1(:)+P)/P;
       err_1   = (tmp'*N*tmp-2*M'*tmp+P)/P;
       
       %Gradient descent update
       if (abs(prev_err_1-err_1)<(10^(-8)))
          if (correct < max_correct)
           tmp = tmp + tau * (N*tmp - M+ alpha * tmp + norm(tmp)*randn(numel(tmp),1));
           correct = correct +1;
          else
              break;
          end
      end
       R_1 = reshape(tmp,M_size(2),M_size(1));
 
       max_R_1 = max(abs(R_1(:)));
      max_R_2 = max(abs(R_2(:)));
      delta_factor = sqrt(max_R_1 *max_R_2);
       
       R_1 = (delta_factor ./ max_R_1) .* R_1;
       R_2 = (delta_factor ./ max_R_2) .* R_2;
       
       temp_B = B; 
       temp_B{n} = tns_mult(temp_B{n},3,R_1,1);
       temp_B{1} = tns_mult(R_2,2,temp_B{1},1);
       
       real_error =TRnorm_diff(Tin,temp_B)/P_sqrt;
       ERR(index_e) = real_error;
       index_e = index_e+1;
       if real_error < tol
           break
       end
       
       %Optimize for the second one:
       N = permute(tns_mult(N_out,3,R_1,1),[1,2,4,3]);
       N = tns_mult(N,4,R_1,1);
       M = permute(tns_mult(M_out,3,R_1,1),[1,2,4,3]);
  
       N_size = size(N);
       N = reshape(permute(N,[3 1 4 2]),N_size(3)*N_size(1),N_size(4)*N_size(2));
       
       M = TRtrace(M,2,4,[1,3]);
       M_size = size(M);
       M = reshape(M',numel(M),1);
    
       tmp = (N+alpha * eye(size(N)))\M;
       prev_err_2 = (R_2(:)'*N*R_2(:) - 2*M'*R_2(:)+P)/P;
       err_2   = (tmp'*N*tmp-2*M'*tmp+P)/P;
       
       %Gradient descent update
       if (abs(prev_err_2-err_2)<(10^(-8))) 
          if (correct < max_correct)
           tmp = tmp + tau * (N*tmp - M+ alpha * tmp + norm(tmp)*randn(numel(tmp),1));
           correct = correct+1;
          else
              break;
          end
       end
      R_2 = reshape(tmp,M_size(2),M_size(1));
      max_R_1 = max(abs(R_1(:)));
      max_R_2 = max(abs(R_2(:)));
      delta_factor = sqrt(max_R_1 *max_R_2);
       
       R_1 = (delta_factor ./ max_R_1) .* R_1;
       R_2 = (delta_factor ./ max_R_2) .* R_2;
            
   end
   
    B{n} = tns_mult(B{n},3,R_1,1);
    self{n}  = permute(tns_mult(B{n},2,B{n},2),[1 3 2 4]);
    cross{n} = permute(tns_mult(B{n},2,Tin{n},2),[1 3 2 4]);
   
    B{1} = tns_mult(R_2,2,B{1},1);
    self{1}  = permute(tns_mult(B{1},2,B{1},2),[1 3 2 4]);
    cross{1} = permute(tns_mult(B{1},2,Tin{1},2),[1 3 2 4]);
      
    shift = circshift((1:n)',idx_order(i)-1);
    cross = cross(shift);  self = self(shift); B = B(shift); Tin = Tin(shift);
    
    if (ii >= n+1) && (ERR(end) < tol)
         break;
     end
    
end
end
