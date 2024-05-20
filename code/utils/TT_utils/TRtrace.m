function X = TRtrace(B,id1,id2,restid)
%computes the trace wrt id1 and id2
B_size = size(B);
B_perm = permute(B,[restid,id1,id2]);
k= prod(B_size(restid));
B_perm = reshape(B_perm,k,B_size(id1),B_size(id2));
 for i=1:k
     X(i)=trace(squeeze(B_perm(i,:,:)));
 end
 X = reshape(X,B_size(restid));
end