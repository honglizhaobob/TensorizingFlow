function [TT] = RLqr(TT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author(s): Yuehaw Khoo
%
%
% Custom implementation of tensor-train QR factorization in 
% right-left sweeping order.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = numel(TT);
for i=fliplr(1:d)
   if i==d
       TT{i} = permute(TT{i},[2, 1]);
       size(TT{i});
       tmp = TT{i}';
       [Q, R] = qr(tmp, 0);
       TT{i} = Q';
       R     = R';   
   elseif i==1
       TT{i} = tns_mult(TT{i},2,R,1);
   else
       TT{i} = permute(TT{i}, [2 1 3]);
       tmp = tns_mult(TT{i},3,R,1);       
       tmp = reshape(tmp,size(TT{i},1),size(TT{i},2)*size(TT{i},3));
       tmp = tmp';
       [Q, R] = qr(tmp, 0);
       TT{i}  = reshape(Q', size(TT{i},1), size(TT{i},2), size(TT{i},3));
       R      = R';
   end
end