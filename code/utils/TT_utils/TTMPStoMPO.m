function [A] = TTMPStoMPO(A)

for i=1:numel(A)
    sz = size(A{i});
    
    
    if i==1
        
        A{i} = reshape(A{i},sqrt(sz(1)),sqrt(sz(1)),sz(2));
        
    elseif i==numel(A)
        
        A{i} = reshape(A{i},sz(1),sqrt(sz(2)),sqrt(sz(2)));
        
    else
        
        A{i} = reshape(A{i},sz(1),sqrt(sz(2)),sqrt(sz(2)),sz(3));
        
    end
    
    
    
end



end