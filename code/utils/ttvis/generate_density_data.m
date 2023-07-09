function [X,Y,Z] = generate_density_data(C,all_A,all_xg,filepath)
    % Used for plotting, given coefficient TT C, 
    % unwrap and save all dimensions. X, Y are only 
    % stored once. Z is (d x grid_size x grid_size) containing
    % density data for all dimensions.
    
    % * this code assumes uniform grid size across all dimensions
    % * data is contained in [-1,1]
    d = C.d;
    if ~iscell(all_A)
        % repeat this data matrix
        tmp = cell(d,1);
        for i = 1:d
            tmp{i} = all_A;
        end
        all_A = tmp;
    end
    
    if ~iscell(all_xg)
        % repeat this grid
        tmp = cell(d,1);
        for i = 1:d
            tmp{i} = all_xg;
        end
        all_xg = tmp;
    end
    
    % grid size
    grid_size = length(all_xg{1});
    % loop over all dimensions and generate data
    Z = zeros([d-1,grid_size,grid_size]);
    for i = 1:d-1
        if i == 1
            [X,Y,tmp] = ftt_visualize(C,all_A,i,all_xg);
        else
            [~,~,tmp] = ftt_visualize(C,all_A,i,all_xg);
        end
        Z(i,:,:) = tmp;
    end
    % save to filepath
    save(filepath,'X','Y','Z');
end