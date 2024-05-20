function I = TRscalar(A,ncore,dim,c)

%Get a scalar TT of same size as A
if ~isempty(A)
    I = cell(1,numel(A));
    for i=1:numel(A)
       I{i} = ones(1,size(A{i},2)); 
    end
    I = TRscale(I,c);
    
else
    I = cell(1,ncore);
    for i=1:ncore
        I{i} = ones(1,dim(i));
    end
end
I = TRscale(I,c);
end 