function [tt_o] = cell2tt(TT)
    % convert cells into TT in TT-Toolbox format
    % TT = (n_i, r_i-1, r_i)
    d = length(TT);
    % preallocate
    new_cores = cell([d,1]);
    for i = 1:d
        if i == 1
            new_cores{i} = reshape(TT{i},[1,size(TT{i})]);
        elseif i == d
            %new_cores{i} = permute(TT{i},[2 1]);
            continue;
        else
            new_cores{i} = permute(TT{i},[2 1 3]);
        end
    end
    tt_o = cell2core(tt_tensor,new_cores);
end
