function [A,Z] = leftsweep_canonical(A)

%Assume A is already a TT, a cell of 3-dim tensor
d = numel(A);
R = cell(1,d);
for i=fliplr(2:d)
    Asize      = size(A{i});

    if i<d
        tmp        = reshape(A{i},Asize(1),prod(Asize(2:3)));
    else
        tmp        = A{i};
    end
    [Q,tmp ]   = qr(tmp',0);
    R{i}       = tmp';

    if i<d
        A{i}       = reshape(Q',size(Q',1), Asize(2), Asize(3));
    else
        if Asize(1)>Asize(2)
            A{i}       = transpose([Q zeros(Asize(2), Asize(1)-Asize(2))]);
        else
            A{i}       = reshape(Q',size(Q',1), Asize(2));
        end
    end

    if i==2
        A{i-1}     = tns_mult(A{i-1},2,R{i},1);
    elseif i==d
        A{i-1}     = tns_mult(A{i-1},3,[R{i} zeros(Asize(1), Asize(1)-Asize(2))],1);
    else
        A{i-1}     = tns_mult(A{i-1},3,R{i},1);
    end

end

%compute norm

Z = sqrt(tns_mult(A{1},[1 2],A{1},[1,2]));

