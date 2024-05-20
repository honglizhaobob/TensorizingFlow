function A = partial_rightsweep_orthogonal(A, idx)

% Assume A is already a TT, a cell of 3-dim tensor
% idx is the index to which we stop the rightsweep
% idx is NOT orthogonalized!

d = numel(A);
R = cell(1,d);
for i=1:idx-1
    Asize      = size(A{i});

    if i>1
        tmp        = reshape(A{i},prod(Asize(1:2)),Asize(3));
    else
        tmp        = A{i};
    end
    [Q,tmp]    = qr(tmp,0);
    R{i}       = tmp;

    if i>1
        A{i}       = reshape(Q, Asize(1), Asize(2), size(Q,2));
    else
        A{i}       = reshape(Q, Asize(1), size(Q,2));
    end
    
    % Update the next node
    A{i+1}     = tns_mult(R{i},2,A{i+1},1);

end


