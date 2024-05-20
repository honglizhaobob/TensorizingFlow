function [I] = TTidentityMPO(n,d)


I = cell(1,n);

for i=1:n
    
    if i==1
        tmp = zeros(d,d,1);
        tmp(:,:,1) = eye(d);
        I{i} = tmp;
        
    elseif i==n
        tmp = zeros(1,d,d);
        tmp(1,:,:) = eye(d);
        I{i} = tmp;
        
    else
        tmp = zeros(1,d,d,1);
        
        tmp(1,:,:,1) = eye(d);
        I{i} = tmp;
    end
    
end


end