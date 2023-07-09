function K = path_energy_kernel_xA(all_points, beta, dt, c)
    % Computes a two-variable matrix representing the kernel
    % (local interaction) between xA and x1, in the
    % non-equilibrium path transition setting.
    % xA core has a slightly different form from common cores.
    % (for xA core)
    %       exp(-0.25 * beta * dt * | (X_1 - X_A) / dt - b(X_1) |^2)
    %       =
    %       exp(-0.25 * (beta / dt) * | ( X_1 - dt * b(X_1) ) - ( X_A ) |^2)
    %   
    %
    % The convention is K(2,1) = p(grid(1), grid(2)) where
    % p is the two variable function the kernel is represeting.
    % 
    % Inputs: all_points,       ((k^2) x 2 array) all possible (col-major) 
    %                           2d points
    %         beta, dt, c       (scalar) parameters of the problem
    %
    % Outputs: 
    %         K,                (2d array) (k^2)x(k^2) matrix re-
    %                           presenting the local PDF.
    %
    % Dependencies:
    %           drift.m
    
    % compute energy for row, then for col, and take norm
    x_vals = all_points(:,1);
    y_vals = all_points(:,2);
    all_drift = noneq_drift(all_points, c);
    drift_x = all_drift(:,1); 
    drift_y = all_drift(:,2);
    % rows are X_i+1, columns are X_i
    energy_x = ( x_vals - dt * drift_x ).' - (x_vals );
    energy_y = ( y_vals - dt * drift_y ).' - (y_vals );
    K = exp(-0.25 * ( beta/dt ) *( energy_x.^2 + energy_y.^2 )); 
end