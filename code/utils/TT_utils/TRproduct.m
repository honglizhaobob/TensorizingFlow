% My other function
function [x] = TRproduct(T,id1,id2)
n = numel(T);
if n < 2
    x = T{1};
else
    x = tns_mult(TRproduct(T(1:int8(n/2)),id1,id2),id1,TRproduct(T((int8(n/2)+1):n),id1,id2),id2);
end
end
