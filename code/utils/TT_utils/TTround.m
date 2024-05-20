function [A] = TTround(A,r)

%Assume A is already a TT, a cell of 3-dim tensor
d     = numel(A);
%delta = delta/sqrt(d-1)*TTnorm(A);

d = numel(A);
R = cell(1,d);
for i=fliplr(2:d)
    Asize      = size(A{i});

    if i<d
        tmp        = reshape(A{i},Asize(1),prod(Asize(2:3)));
    else
        tmp        = A{i};
    end
    [Q,tmp ]   = qr(tmp',0);
    R{i}       = tmp';

    if i<d
        A{i}       = reshape(Q',size(Q',1), Asize(2), Asize(3));
    else
        A{i}       = reshape(Q',size(Q',1), Asize(2));
    end

    if i>2
        A{i-1}     = tns_mult(A{i-1},3,R{i},1);
    else
        A{i-1}     = tns_mult(A{i-1},2,R{i},1);
    end

end

for i=1:d-1

   if i>1&&ndims(A{i}) == 2
       break;
   end
   
   if i==14
       0;
   end

   Asize       = size(A{i});
   if i>1
       tmp = reshape(A{i},prod(Asize(1:2)),Asize(3));
   else
       tmp = A{i};
   end

   [U,sigma,V] = svd(tmp);


   idx         = 1:r;


   if i==1
       if r>size(U,2)
           tmp  = U*sigma(:,idx);
           A{i} = reshape(tmp,Asize(1),r);
           A{i+1}      = tns_mult(V(:,idx)',2,A{i+1},1);
       else
           A{i}        = reshape(U(:,idx),Asize(1),r);
           A{i+1}      = tns_mult(sigma(idx,idx)*V(:,idx)',2,A{i+1},1);
       end
   else
       A{i}        = reshape(U(:,idx),Asize(1),Asize(2),r);
       A{i+1}      = tns_mult(sigma(idx,idx)*V(:,idx)',2,A{i+1},1);
   end

end
