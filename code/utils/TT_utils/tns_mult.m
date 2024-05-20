function Z = tns_mult(A,aid, B,bid)
%multiply tensor A and B using dim aid and dim bid, with diag matrix mid sandwiched in the middle


    nA = ndims(A);
    nB = ndims(B);


tmpa = zeros(1,nA);
tmpa(aid) = 1;
are = sort(find(tmpa==0),'ascend');

tmpb = zeros(1,nB);
tmpb(bid) = 1;
bre = sort(find(tmpb==0),'ascend');

A = permute(A, [are, aid]);
nta = size(A);
nta1 = nta(1:numel(are));    nta2 = nta((numel(are)+1):end);
A = reshape(A, prod(nta1),prod(nta2));

B = permute(B, [bid, bre]);
ntb = size(B);
ntb1 = ntb(1:numel(bid));    ntb2 = ntb((numel(bid)+1):end);
B = reshape(B, prod(ntb1),prod(ntb2));

%Z = full(A*sparse(diag(mid))*B);
Z = A*B;

if ~isempty(nta1)
    Z = reshape(Z, [nta1 ntb2]);
end
end

