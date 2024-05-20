function I = TReye(A)
%Get an identity TT of same size as A
if ~isempty(A)
    I = cell(1,numel(A));
    for i=1:numel(A)
        I{i} = zeros(size(A{i}));
        for j = 1:size(A{i},2)
            I{i}(:,j,:)=eye(size(A{i},1),size(A{i},3)); 
        end
    end
end 