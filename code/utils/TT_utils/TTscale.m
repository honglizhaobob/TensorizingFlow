function [A] = TTscale(A,scale)
%Scaling tensor train
%A{1} = scale*A{1};

% Now scale each block equally
d         = numel(A);
tmp_scale = scale^(1/d);
for i = 1:d
    A{i} = tmp_scale * A{i};
end
