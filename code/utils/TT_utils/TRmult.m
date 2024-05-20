function [C] = TRmult(A,B)
%Multiplying two tensor ring (periodic)
%It groups it using the kronecker product (outer product)
C = cell(numel(A),1);

for i=1:numel(A)
    
    tmp = zeros(size(A{i},1)*size(B{i},1),size(A{i},2),size(A{i},3)*size(B{i},3));
    n   = size(A{i},2);
    
    for k=1:n
        tmp(:,k,:) = kron(squeeze(A{i}(:,k,:)),squeeze(B{i}(:,k,:)));
    end
    C{i} = tmp;
end
end


% if size(C{1},1)<size(C{1},2)
%     [Q, R] = qr(C{1},0); 
%     C{1}   = Q;
%     C{2}   = tns_mult(R,2,C{2},1);
% end
% 
% 
% % for i=2:numel(C)-1
% %     Csize = size(C);
% %     tmp   = rehsape(C{i},prod(Csize(1:2)),Csize(3));
% %     [~,sigma,V] = svd(tmp);
% %     idx = find(diag(sigma)>=10^-12);
% %     
% %     [Q,R] = qr(tmp,0);
% %     C{i}  = reshape(C{i},2
% %     C{i}  = reshape(Q,3,3);
% %     C{i}  = 
% % end
% 
% if size(C{end},2)<size(C{end},1)
%     [Q, R] = qr(C{end}',0); 
%     C{end}     = Q';
%     C{end-1}   = tns_mult(C{end-1},3,R',1);
% end
