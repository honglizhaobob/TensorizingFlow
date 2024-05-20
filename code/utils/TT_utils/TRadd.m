function C = TRadd(A,B)
%Adding two tensor ring (periodic)
for i=1:numel(A)
    %Just remove the singularity in the corners 
    %The trace holds in the large matrix:
    tmp = zeros(size(A{i},1)+size(B{i},1),size(A{i},2),size(A{i},3)+size(B{i},3));
    
    for k=1:size(A{i},2)
        tmp(:,k,:) = blkdiag(squeeze(A{i}(:,k,:)),reshape(squeeze(B{i}(:,k,:)),size(B{i},1),size(B{i},3)));
         
    end 
    C{i} = tmp;
end

% 
% if size(C{1},1)<size(C{1},2)
%     [Q, R] = qr(C{1}); 
%     C{1}   = Q;
%     C{2}   = tns_mult(R,2,C{2},1);
% end
% 
% if size(C{end},2)<size(C{end},1)
%     [Q, R] = qr(C{end}'); 
%     C{end}     = Q';
%     C{end-1}   = tns_mult(C{end-1},3,R',1);
% end
% 
