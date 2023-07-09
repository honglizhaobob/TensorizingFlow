function yes = isin_grid(pnt, grid)
    % helper function to check if a point PNT is in an 2d GRID.
    %
    %
    %   pnt,            (1 x 2 array) the point (x, y) to check in 
    %                   GRID x GRID
    %   grid,           (array)       grid points
    %
    %   yes,            (bool)        status
    if length(pnt) ~= 2
        error("> Input Error: point must be in 2d. ");
    end
    x = pnt(1); y = pnt(2);
    yes = false;
    if ismember(x, grid) && ismember(y, grid)
        yes = true;
    end
end