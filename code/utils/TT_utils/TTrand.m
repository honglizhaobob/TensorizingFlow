function I = TTrand(ncore,dim,r)



I = cell(1,ncore);
for i=1:ncore
    if i==1
        I{i} = randn(dim(i),r);
    elseif i==ncore
        I{i} = randn(r,dim(i));
    elseif i>1&&i<ncore
        I{i} = randn(r,dim(i),r);
    end
    
end


