function A = rightsweep_current(A, curr_idx)

% Assume A is already a TT, a cell of 3-dim tensor
% curr_idx is the index to orthogonalize using a right-sweep

d = numel(A);

if curr_idx == d
    A = leftsweep_canonical(A);
else
    Asize      = size(A{curr_idx});
    
    if curr_idx>1
        tmp        = reshape(A{curr_idx},prod(Asize(1:2)),Asize(3));
    else
        tmp        = A{curr_idx};
    end
    [Q,tmp]    = qr(tmp,0);

    if curr_idx>1
        A{curr_idx}       = reshape(Q, Asize(1), Asize(2), size(Q,2));
    else
        A{curr_idx}       = reshape(Q, Asize(1), size(Q,2));
    end

    % Update the next node
    A{curr_idx+1}     = tns_mult(tmp,2,A{curr_idx+1},1);
end
end
