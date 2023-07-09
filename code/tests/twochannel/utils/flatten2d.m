function [all_idx, row_indexer, col_indexer, row_coord, col_coord] = ...
    flatten2d(grid)
    % Returns all necessary profiles of a uniform 2d grid
    % (namely, GRID x GRID is our 2d domain). Assumes column
    % major flattening. The flattening is a column-major enumeration
    % of all possible points in 2d, specified by GRID.
    %
    % Note:
    %       Same as MATLAB: [X_grid, Y_grid] = meshgrid(x_grid);
    %                       [X_grid(:), Y_grid(:)]
    %
    %
    %
    %
    %  Outputs:
    %           all_idx,                (array) 1:(k^2) where k is
    %                                   number of points on GRID
    %           row/col_indexer,        (array) the row and col
    %                                   indices in each direction. 
    %                                   A point can be retrieved by
    %                                   [ grid(row_indexer),
    %                                   grid(col_indexer) ]
    %           row/col_coord,          (array) actual coordinates
    %                                   given by indexers.
    
    % end points
    x_start = grid(1); x_end = grid(end);
    dx = grid(2)-grid(1);
    k = length(grid);
    idx = 1:(k^2);
    % column major indexer into 2d grid [-L,L]x[-L,L]
    row_indexer = ceil(idx / k); 
    col_indexer = mod(idx-1, k) + 1;
    % convert indexer into coordinates
    row_coord = x_start + (row_indexer - 1) * dx;
    col_coord = x_start + (col_indexer - 1) * dx;
    all_idx = idx;
end