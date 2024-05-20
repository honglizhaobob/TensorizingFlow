function B = TTln(A, nlevel, newton_iter, newton_acc, newton_scale, round_acc)
%log using newton iteration+continue fraction

%Compute log(1+A)
d = numel(A);
I = cell(1,d);
for i=1:d
    if i>1&&i<d
        
        I{i} = ones(1,size(A{i},2),1);
        
    elseif i==1
        
        I{i} = ones(size(A{i},1),1);
        
    elseif i==numel(A)
        
        I{i} = ones(1,size(A{i},2));
        
    end
end

a = nlevel;
b = (floor((nlevel-1)/2)+1)^2;
B = TTadd(TTscale(I,a),TTscale(A,b/(a+1)));
%TTeval(A,ones(1,numel(B)))/TTeval(B,ones(1,numel(B)))


for i=flip(1:nlevel-1)
 
    a = i;
    b = (floor((i-1)/2)+1)^2;
    
    B = TTadd(TTscale(I,a),TTmult(TTscale(A,b),TTreciprocal_newton(B,newton_iter, newton_scale, newton_acc)));
    
    B = TTround(B,round_acc); 
end
tmp = TTround(TTreciprocal_newton(B,newton_iter, newton_scale, newton_acc), round_acc);
B   = TTround(TTmult(A,tmp), round_acc);
