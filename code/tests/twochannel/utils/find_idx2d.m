function [xpos, ypos] = find_idx2d(pnt, grid)
    % Given a point in 2d, whose grid is assumed to be
    % GRID x GRID. Finds the multi-index of the point from 
    % the (GRID x GRID) is flattened, column-major.
    % Also includes a check that the point in grid 
    % using (xpos, ypos) is in fact the orginal PNT.
    %
    %
    % Inputs:
    %       pnt,                (1 x 2) 2d point for which
    %                           we would like to locate in
    %                           (GRID x GRID)
    %       grid,               (1d array) discretized grid for
    %                           each direction.
    %
    %       xpos, ypos,         (scalar) multi-index in the range
    %                           of 1:k, k = length(GRID)
    assert(length(pnt) == 2, "> Point must be in 2d. ");
    k = length(grid); 
    x_start = grid(1);
    y_start = x_start;
    % step size
    dx = grid(2)-grid(1);
    dy = dx;
    pnt_x = pnt(1); 
    pnt_y = pnt(2); 
    xpos = round((pnt_x - x_start)/dx) + 1;
    ypos = round((pnt_y - y_start)/dy) + 1;
    % post check if indices were found correctly
    assert(all([grid(xpos), grid(ypos)] == pnt), ...
        "> multi-index is incorrect. ");
end