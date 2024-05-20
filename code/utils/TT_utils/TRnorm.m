function normA = TRnorm(A)
n = numel(A);
tmp = cell(1,n); 
for i=1:n
    tmp_i = tns_mult(A{i},2,A{i},2);
    tmp{i} = permute(tmp_i,[1 3 2 4]);
end
M1 = TRproduct(tmp(1:int8(n/2)),[3,4],[1,2]);
M2 = TRproduct(tmp((int8(n/2)+1):n),[3,4],[1,2]);
normA = M1(:)'*reshape(permute(M2,[3 4 1 2]),numel(M2),1);
normA = sqrt(abs(normA));
end