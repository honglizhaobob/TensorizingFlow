function A = partial_leftsweep_orthogonal(A, idx)

% Assume A is already a TT, a cell of 3-dim tensor
% idx is the index to which we stop the leftsweep
% idx is NOT orthogonalized!

d = numel(A);
R = cell(1,d);
for i=fliplr(idx+1:d)
    Asize      = size(A{i});

    if i<d
        tmp        = reshape(A{i},Asize(1),prod(Asize(2:3)));
    else
        tmp        = A{i};
    end
    [Q,tmp]    = qr(tmp',0);
    R{i}       = tmp';

    if i<d
        A{i}       = reshape(Q',size(Q',1), Asize(2), Asize(3));
    else
        A{i}       = reshape(Q',size(Q',1), Asize(2));
    end

    if i>2
        A{i-1}     = tns_mult(A{i-1},3,R{i},1);
    else
        A{i-1}     = tns_mult(A{i-1},2,R{i},1);
    end

end


