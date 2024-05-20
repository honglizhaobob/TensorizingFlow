function normA = TTnorm(A)

normA = tns_mult(A{1},1,A{1},1);

for i=2:numel(A)
    tmp  = tns_mult(A{i},2,A{i},2);
    if i<numel(A)
        tmp  = permute(tmp,[1 3 2 4]);
    end
    %2-ndims(normA)
    nA       = 2;%ndims(normA);   
    normA    = tns_mult(normA, nA-1:nA, tmp, [1 2]);
end

normA = sqrt(normA);