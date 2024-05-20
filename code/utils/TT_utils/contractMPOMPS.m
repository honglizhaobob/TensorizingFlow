function [L,R] = contractMPOMPS(O,S,I,idx)

% idx: stopping index for contraction
% If size(idx) = [2,1], perform left contraction up to idx(1) and right
% contraction up to idx(2). Need idx(2)>idx(1)

if numel(idx) == 1
    %contract each node
    n = numel(O);

    L = 1;
    for i=1:idx-1

        L = L*I{i};

    end
    if idx>1
        
        L = reshape(L,size(S{idx},1),size(O{idx},1),size(S{idx},1));
    end

    R = 1;

    for i=fliplr(idx+1:n)
        %i
        %size(I{i})
        R = I{i}*R;

    end

    if idx<n
        if idx==1
            R = reshape(R,size(S{idx},2),size(O{idx},3),size(S{idx},2));
        else
            R = reshape(R,size(S{idx},3),size(O{idx},4),size(S{idx},3));
        end

    end
else
    % Check
    if idx(2) - idx(1) ~= 1
        error('Check indices!')
    end
    
    %contract each node
    n = numel(O);

    L = 1;
    for i=1:idx(1)-1

        L = L*I{i};

    end
    if idx(1)>1
        L = reshape(L,size(S{idx(1)},1),size(O{idx(1)},1),size(S{idx(1)},1));
    end

    R = 1;

    for i=fliplr(idx(2)+1:n)
        %i
        %size(I{i})
        R = I{i}*R;

    end

    if idx(2)<n
        if idx(2)==1
            R = reshape(R,size(S{idx(2)},2),size(O{idx(2)},3),size(S{idx(2)},2));
        else
            R = reshape(R,size(S{idx(2)},3),size(O{idx(2)},4),size(S{idx(2)},3));
        end

    end

end
