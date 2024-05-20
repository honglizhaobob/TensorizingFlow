function [red_A] = TRround(A,delta)

%Assume A is already a TR, a cell of 3-dim tensor
%We will use the heuristics of cutting of the degree 
%Step by step, removing one node and rounding the other train
r = 0;
red_A = A;
d = numel(A);
niter = 10 *d;
idx_order = randsample(d,niter,true);

for i = 1 : niter
    shift = circshift((1:d)',-(idx_order(i)-1));
    red_A = red_A(shift);   

    train = red_A(2:d);
    numel(train);
    if d >2 
        Tsize_d = size(train{d-1})
        train{d-1} = reshape(train{d-1}, Tsize_d(1),prod(Tsize_d(2:3))); 
    end
    if d > 1
        Tsize_1 = size(train{1})
        train{1} = reshape(train{1}, prod(Tsize_1(1:2)),Tsize_1(3)); 
        [train,r] = TTround(train,delta);
        Asize_1 = size(train{1});
        train{1} = reshape(train{1}, Tsize_1(1),Tsize_1(2),Asize_1(2)); 
    end
    if d >2 
        Asize_d = size(train{d-1});
        train{d-1} = reshape(train{d-1}, Asize_d(1),Tsize_d(2),Tsize_d(3)); 
    end
 
    red_A(2:d) = train;    
    shift = circshift((1:d)',idx_order(i));
    red_A = red_A(shift);


    i
    r
    TRnorm(TRsubtract(A,red_A)) 

end
end
