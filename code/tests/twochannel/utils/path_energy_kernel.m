function K = path_energy_kernel(all_points, beta, dt, c)
    % Computes a two-variable matrix representing the kernel
    % (local interaction) between the two variables, in the
    % non-equilibrium path transition setting.
    % (for common core)
    %       exp(-0.25 * beta * dt * | (X_i+1 - X_i) / dt - b(X_i) |^2)
    %       =
    %       exp(-0.25 * (beta / dt) * | ( X_i+1 ) - ( X_i + dt * b(X_i) ) |^2)
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
    energy_x = x_vals.' - (x_vals + dt * drift_x);
    energy_y = y_vals.' - (y_vals + dt * drift_y);
    K = exp(-0.25 * ( beta/dt ) *( energy_x.^2 + energy_y.^2 )); 
end