function [A] = right_canonical(A,ismpo,idx)


if ~ismpo
    sz = size(A{idx});
    if idx~=1
        tmp   = reshape(A{idx},sz(1)*sz(2),sz(3));
        [Q,R] = qr(tmp,0);
        
        A{idx} = reshape(Q,sz(1),sz(2),sz(3));
        A{idx+1} = tns_mult(R,2,A{idx+1},1);
    else
        
        tmp = A{idx};
        [Q,R] =  qr(tmp,0);
        
        A{idx}   = Q;
        A{idx+1} = tns_mult(R,2,A{idx+1},1);
    end
else

    sz = size(A{idx});


    if idx~=1
        tmp   = reshape(A{idx},sz(1)*sz(2)*sz(3),sz(4));
        [Q,R] = qr(tmp,0);
        
        A{idx}   = reshape(Q,sz(1),sz(2),sz(3),sz(4));
        A{idx+1} = tns_mult(R,2,A{idx+1},1);
    else
        
        tmp   = reshape(A{idx},sz(1)*sz(2),sz(3));
        [Q,R] = qr(tmp,0);
        
        A{idx}   = reshape(Q,sz(1),sz(2),sz(3));
        A{idx+1} = tns_mult(R,2,A{idx+1},1);
    end


end

end