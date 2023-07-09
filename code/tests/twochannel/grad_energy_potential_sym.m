function grad_S_sym = grad_energy_potential_sym(N, dt, beta)
    % (symbolic) computes the exact gradient of the energy 
    % potential, returns an array of function handles that
    % are dS/dx_i for each i = 1, 2, ..., N
    
    % Inputs: 
    %       N,                  (scalar) number of time discretization
    %       dt,                 (scalar) time step size
    %       beta,               (scalar) parameter, inverse temperature
    
    % Outputs:
    %       grad_S_sym          (function handle) a function that takes in 
    %                           U(:) and outputs Nx2 gradients at each
    %                           path point
    
    % get symbolic energy function
    U = sym('u', [N, 2], 'real');
    S = energy_potential(U(:), dt, beta);
    % get gradient, length (N)
    grad_S = sym(zeros(N, 2));
    for i = 1:length(U)
        % differentiate with respect to x_i
        grad_S(i, :) = gradient(S, U(i, :));
    end
    % convert into function handle that depends on
    % x1, x2, ..., xN and returns N x 2 matrix
    grad_S_sym = matlabFunction(grad_S, 'Vars', U(:));
end