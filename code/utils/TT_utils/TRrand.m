function I = TRrand(A,ncore,dim,r,c)

%Get a scalar TR of same size as A
if ~isempty(A)
    I = cell(1,numel(A));
    for i=1:numel(A)
      
            I{i} = c*rand(r,size(A{i},2),r);
   
        
    end
    
else
    
    I = cell(1,ncore);
    for i=1:ncore
        if i==1
            I{i} = c*randn(r,dim(i),r);
        elseif i==ncore
            I{i} = randn(r,dim(i),r);
        elseif i>1&&i<ncore
            I{i} = randn(r,dim(i),r);
        end
        
    end
    
    
end