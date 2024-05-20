function B = TRround_2(B,r_target)

stmp  = size(B{1});
r     = stmp(1);   n = stmp(2);
d     = numel(B);
BT    = B;

BT{1} = reshape(B{1},r*n,r);
BT{d} = reshape(B{d},r,r*n);

BT    = TTround(BT,r_target);
B     = BT;
B{1}  = reshape(B{1},r,n,r_target);
B{d}  = reshape(B{d},r_target,n,r);

tmp   = tns_mult(B{d},3,B{1},1);
tmp   = reshape(tmp,r_target*n,r_target*n);

[U,sigma,V] = svd(tmp);
B{d}        = reshape(U(:,1:r_target)*sqrt(sigma(1:r_target,1:r_target)),r_target,n,r_target);
B{1}        = reshape(sqrt(sigma(1:r_target,1:r_target))*V(:,1:r_target)',r_target,n,r_target);
