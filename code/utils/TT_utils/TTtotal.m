function [Z] = TTtotal(A)

%Compute the sum of all tensor entries

Z = squeeze(tns_mult(A{1},1,ones(size(A{1},1),1),1))';
for i=2:numel(A)
    Z = Z*squeeze(tns_mult(A{i},2,ones(size(A{i},2),1),1));
    
    
end

