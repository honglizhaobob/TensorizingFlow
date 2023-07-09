function [x,y,z] = ftt_visualize(C,A,i,all_xg)
    % visualizes a continuous tensor train
    % expansion in the (i-1, i) plane. Assume
    % Legendre grid in every dimension
    
    % C      coefficient train
    % A      spatial grid, cell array of
    %        legendre grid data
    % i      (i >= 2) which plane to plot
    
    % number of total dimensions
    d = C.d;
    assert((d<=C.d) && (d>=2));
    % rebuild TT density
    all_cores = core(C);
    for n = 1:d
        if n == 1
            tmp = tns_mult(all_cores{n},1,A{n},1)';
            all_cores{n} = reshape(tmp, [1 size(tmp)]);
        elseif n == d
            all_cores{n} = tns_mult(all_cores{n},1,A{n},1);
        else
            all_cores{n} = permute(tns_mult(all_cores{n},1,A{n},1),[1 3 2]);
        end
    end

    % contract all other cores with spatial grid to get marginal
    for n = 1:d
        if (n ~= i) && (n ~= i-1)
            % integrate out variable x_i
            all_cores{n} = tns_mult(all_cores{n},2,all_xg{n},2);
        end
    end
    
    % contract all other cores with each other to build marginal
    
    % front 
    tmp1 = all_cores{1};
    for n = 2:i-1
        tmp1 = squeeze(tns_mult(tmp1,2,all_cores{n},1));
    end

    % back
    tmp2 = all_cores{end};
    for n = d-1:-1:i
        tmp2 = tns_mult(all_cores{n},length(size(all_cores{n})),tmp2,1);
    end
    
    % rebuild marginal and plot
    z = tmp1 * tmp2;
    % meshgrid
    [y,x] = meshgrid(all_xg{i-1}',all_xg{i}');
end