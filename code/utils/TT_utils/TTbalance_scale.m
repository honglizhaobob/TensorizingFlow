function [A,new_scale,tmp_norm] = TTbalance_scale(A,curr_scale,idx)

% Balance the scaling of a TT
% idx: the node which is imbalanced now

d = numel(A);
n = numel(idx);

tmp_norm = zeros(n,1);
for i = 1:n
    tmp_size = size(A{idx(i)});
    if numel(tmp_size) == 3
        tmp_norm(i) = norm(reshape(A{idx(i)}, tmp_size(1), prod(tmp_size(2:3))), 'fro');
    else
        tmp_norm(i) = norm(A{idx(i)}, 'fro');
    end
end

% Compute the new scale
new_scale = (prod(tmp_norm))^(1/d) * curr_scale^((d-n)/d);

% Now scale each block equally
id = 1;
for i = 1:d
    if ismember(i,idx)
        A{i} = (new_scale/tmp_norm(id))  * A{i};
        id   = id + 1;
    else
        A{i} = (new_scale/curr_scale)    * A{i};
    end
end
