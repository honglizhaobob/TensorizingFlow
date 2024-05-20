function [A] = TRinitial(A,r)
%Truncates A{i} using SVD for the first r singular values
n = numel(A);
for i = 1:n
    tmp = A{i};
    tmp_size = size(tmp);
    [U,S,V] = svd(reshape(tmp,prod(tmp_size(1:2)),tmp_size(3)));
    A{i} = reshape(U(:,1:r),tmp_size(1),tmp_size(2),r);
    if i < n
        A{i+1} = tns_mult(S(1:r,1:r)*V(:,1:r)',2,A{i+1},1);
    end
    if i > (n-1)
        A{1} = tns_mult(S(1:r,1:r)*V(:,1:r)',2,A{1},1);
    end
end
end