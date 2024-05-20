function C = TTadd(A,B)
%Adding two tensor train (non-periodic)
for i=1:numel(A)
    
    
    if i~=1&&i~=numel(A)
        tmp = zeros(size(A{i},1)+size(B{i},1),size(A{i},2),size(A{i},3)+size(B{i},3));

        for k=1:size(A{i},2)
            

            tmp(:,k,:) = blkdiag(squeeze(A{i}(:,k,:)),reshape(squeeze(B{i}(:,k,:)),size(B{i},1),size(B{i},3)));
         
        end
    elseif i==1
        tmp = zeros(size(A{i},1),size(A{i},2)+size(B{i},2));
        for k=1:size(A{i},1)
            tmp(k,:)    = [A{i}(k,:) B{i}(k,:)];
        end
    elseif i==numel(A)
        tmp = zeros(size(A{i},1)+size(B{i},1),size(A{i},2));
        for k=1:size(A{i},2)
            tmp(:,k)    = [A{i}(:,k); B{i}(:,k)];
        end
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
