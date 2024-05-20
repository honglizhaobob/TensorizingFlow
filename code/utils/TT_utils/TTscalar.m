function I = TTscalar(A,ncore,dim,c)

%Get a scalar TT of same size as A
if ~isempty(A)
    I = cell(1,numel(A));
    for i=1:numel(A)
        if i==1
            I{i} = c*ones(size(A{i},1),1);
        elseif i==numel(A)
            I{i} = ones(1,size(A{i},2),1);
        elseif i>1&&i<numel(A)
            I{i} = ones(1,size(A{i},2));
        end
        
    end
    
else
    
    I = cell(1,ncore);
    for i=1:ncore
        if i==1
            I{i} = c*ones(dim(i),1);
        elseif i==ncore
            I{i} = ones(1,dim(i),1);
        elseif i>1&&i<ncore
            I{i} = ones(1,dim(i));
        end
        
    end
    
    
    
end