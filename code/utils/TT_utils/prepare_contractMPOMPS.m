function [I,tot] = prepare_contractMPOMPS(O,S,idx)

%idx: nodes that want to be prepared for contraction

%contract each node
n = numel(O);

I = cell(1,n);
tot = 1;
for i=idx
    if i==1
        
        %         tmp  = tns_mult(S{i},1,O{i},1);
        %         tmp  = permute(tmp,[2 1 3]);
        %         tmp  = reshape(tmp,size(O{i},2),size(O{i},3)*size(S{i},2));
        %
        %         I{i} = reshape(tns_mult(tmp,1,S{i},1),1,size(O{i},3)*size(S{i},2)^2);
        
        tmp  = tns_mult(S{i},1,O{i},1);
        tmp  = tns_mult(tmp,2,S{i},1);
        I{i} = reshape(tmp,1,size(S{i},2)^2*size(O{i},3));
        
    elseif i==n
        
        
        %         tmp  = tns_mult(S{i},2,O{i},2);
        %         tmp  = reshape(tmp,size(S{i},1)*size(O{i},1),size(O{i},3));
        %         I{i} = reshape(tns_mult(tmp,2,S{i},2),size(O{i},1)*size(S{i},1)^2,1);
        
        tmp  = tns_mult(S{i},2,O{i},2);
        tmp  = tns_mult(tmp,3,S{i},2);
        I{i} = reshape(tmp,size(S{i},1)^2*size(O{i},1),1);
        
        
    else
        
        %
        %         tmp = tns_mult(S{i},2,O{i},2);
        %         tmp = permute(tmp,[1 3 4 2 5]);
        %         tmp = reshape(tmp,size(S{i},1)*size(O{i},1),size(O{i},3), size(S{i},3)*size(O{i},4));
        %
        %         tmp = tns_mult(tmp,2,S{i},2);
        %         K = reshape(permute(tmp,[1 3 2 4]),size(S{i},1)^2*size(O{i},1),size(S{i},3)^2*size(O{i},4));
        %
        
        if ndims(O{i})==4
            
            tmp = tns_mult(S{i},2,O{i},2);
            tmp = tns_mult(tmp,4,S{i},2);
            I{i} = reshape(permute(tmp,[1 3 5 2 4 6]),size(S{i},1)^2*size(O{i},1),size(S{i},3)^2*size(O{i},4));
            
        else
            
            tmp = tns_mult(S{i},2,O{i},2);
            tmp = tns_mult(tmp,4,S{i},2);
            I{i} = reshape(permute(tmp,[1 3 4 2 5]),size(S{i},1)^2*size(O{i},1),size(S{i},3)^2*size(O{i},4));
            
        end
        
        %        norm(K-I{i})
    end
    
    tot = tot*I{i};
end




end