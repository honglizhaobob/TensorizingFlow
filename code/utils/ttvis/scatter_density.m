function values = scatter_density(data1,data2,resolutionx,resolutiony)
    % generate density heatmap from discrete data 
    % points. Good to show FTT's accuracy.
    % data1, data2 need to have same length
    % and are row vectors of size (1 x size)

    % Assumes [-1,1] hypercube grid.
    values = hist3([data1(:) data2(:)],[resolutionx,resolutiony]);
    x_grid = linspace(-1,1,resolutionx);
    y_grid = linspace(-1,1,resolutiony);
    pcolor(x_grid,y_grid,values.'); shading interp;
end