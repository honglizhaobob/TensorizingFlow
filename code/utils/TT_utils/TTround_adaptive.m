function [A] = TTround_adaptive(A, mode, svd_thresh, relax)

%Assume A is already a TT, a cell of 3-dim tensor
%delta = delta/sqrt(d-1)*TTnorm(A);

d = numel(A);
R = cell(1,d);

% Using the F nomr of A to define the threshold
% Only works when mode == 'thresh-by-norm'
if strcmp(mode , 'thresh-by-norm')
    svd_thresh = svd_thresh*TTnorm(A)/sqrt(d-1);
end


% Right-to-left QR orthogonalization
for i=fliplr(2:d)
    Asize      = size(A{i});

    if i<d
        tmp        = reshape(A{i},Asize(1),prod(Asize(2:3)));
    else
        tmp        = A{i};
    end
    [Q,tmp ]   = qr(tmp',0);
    R{i}       = tmp';

    if i<d
        A{i}       = reshape(Q',size(Q',1), Asize(2), Asize(3));
    else
        A{i}       = reshape(Q',size(Q',1), Asize(2));
    end

    if i>2
        A{i-1}     = tns_mult(A{i-1},3,R{i},1);
    else
        A{i-1}     = tns_mult(A{i-1},2,R{i},1);
    end

end

% Left-to-right SVD truncation
for i=1:d-1

   Asize       = size(A{i});
   if i>1
       tmp = reshape(A{i},prod(Asize(1:2)),Asize(3));
   else
       tmp = A{i};
   end

   [U,S,V] = svd(tmp,'econ');


   if strcmp(mode , 'thresh') || strcmp(mode , 'thresh-by-norm')
        % Select the singular values > svd_thresh
        idx        = diag(S)>svd_thresh;
        % if not singular value>svd_thresh, keep the largest one
        if sum(idx) == 0
            idx = 1;
        end
        % Relaxation
        tmp_idx    = 1:size(U,2);
        idx        = tmp_idx(idx);
        if numel(idx)<=size(U,2)-relax
            idx    = [idx idx+1:idx+relax];
        end
    elseif strcmp(mode , 'percent') 
        total_var  = sum(diag(S));
        tmp_thresh = total_var*svd_thresh;
        idx        = find(cumsum(diag(S))>=tmp_thresh,1,'first');
        % Relaxation
        idx        = 1:idx;
        tmp_idx    = 1:size(U,2);
        idx        = tmp_idx(idx);
        if numel(idx)<=size(U,2)-relax
            idx    = [idx idx+1:idx+relax];
        end
    else
        error('Mode not implemented!')
   end
   disp(idx(end))

    % Modify the next node
    % Reshape current node
    if i>1
        A{i}       = reshape(U(:,idx), Asize(1), Asize(2), size(U(:,idx),2));
    else
        A{i}       = reshape(U(:,idx), Asize(1), size(U(:,idx),2));
    end
    
    % Update the next node
    A{i+1}     = tns_mult(S(idx,idx)*(V(:,idx).'),2,A{i+1},1);

end
