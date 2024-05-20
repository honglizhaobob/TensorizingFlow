function Y = TTreciprocal_newton(A,niter, scale, acc)

%Element-wise reciprocal via Newton
%scale: constant for initialization


Y = cell(1,numel(A));

d = numel(A);

for i=1:numel(A)
    if i>1&&i<numel(A)
    
        Y{i} = ones(1,size(A{i},2),1);

    elseif i==1
      
        Y{i} = scale*ones(size(A{i},1),1)/d;

    elseif i==numel(A)

        Y{i} = ones(1,size(A{i},2));

    end
   
end


for i=1:niter
    
    tmp  = TTmult(Y,Y);
    if i>1
       tmp = TTround(tmp, acc);
    end
    tmp  = TTround(TTmult(A,tmp), acc);
    
    Y  = TTsubtract(TTscale(Y,2),tmp);
   
    Y  = TTround(Y, acc);
    
end


