function S =  energy_potential_brownian_bridge(path, dt, beta)
    % Energy potential of the Brownian Bridge process
    % with endpoint conditioning. The induced probability
    % density is absolutely continuous with the nonequilibrium
    % path transition.
    
    % Inputs:
    %       path,           (2N x 1) vector describing the path in 2d
    %                       which is originally (N x 2) and used X(:)
    %                       i.e. 
    %                       X    =    [x1_1, x1_2]
    %                                 [x2_1, x2_2]
    %                                 ...    ...
    %                                 [xN_1, xN_2]
    %
    %
    %       dt,             (scalar) time step
    %       beta,           (scalar) inverse temperature parameter
    
    % Output:
    %       S,              (scalar) energy of the sample path
    
    X = reshape(path, [], 2);
    % boundary points
    xA = [-1,(0-5/6)]./3; xB = [1,(0-5/6)]./3;
    
    % number of nodes
    N = size(X, 1);
    S = 0;
    if N > 1
        S_vec = (1/dt)*(X(2:N, :) - X(1:(N-1), :));
        S = sum(sum(S_vec.^2));
    end
    % add penalty from boundary
    S = S + sum( ( (1/dt)*(X(1, :)-xA) ).^2 );
    S = S + sum( ( (1/dt)*(xB-X(end, :)) ).^2 );
    S = beta * dt * S;
end