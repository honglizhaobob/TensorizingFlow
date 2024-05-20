function [A] = TTMPOtoMPS(A)

for i=1:numel(A)
    sz = size(A{i});
    
    
    if i==1
        
        A{i} = reshape(A{i},sz(1)*sz(2),sz(3));
        
    elseif i==numel(A)
        
        A{i} = reshape(A{i},sz(1),sz(2)*sz(3));
        
    else
        
        A{i} = reshape(A{i},sz(1),sz(2)*sz(3),sz(4));
        
    end
    
    
    
end



end