function S = energy_potential(path, dt, beta)
    % discretized evaluation of path integral in 
    % the Brownian bridge, which is a solution to
    %
    %   dXt = b(Xt)dt + sqrt(2/beta)dWt
    %
    % this function includes the nonequilibrium
    % drift choice used in the original paper
    
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
    c = 0; % =2.5, if =0, no nonequilibrium term
    % boundary points
    xA = [-1, 0]; xB = [1, 0];
    
    % number of nodes
    N = size(X, 1);
    S = 0;
    
    forward_difference = false; % whether use forward differencing for path integral
    if forward_difference
        % original: \int |dX/dt - b(x)|^2 dt 
        %           \approx dt * \sum_i |(X_i+1 - X_i)/dt - b(X_i)|^2
    
        if N > 1
            S_vec = (1/dt)*(X(2:N, :) - X(1:(N-1), :)) - ...
                noneq_drift(X(1:(N-1), :), c);
            S = sum(sum(S_vec.^2));
        end
    
        % add penalty from boundary
        S = S + sum( ( (1/dt)*(X(1, :)-xA) - noneq_drift(X(1,:) ,c) ).^2 );
        S = S + sum( ( (1/dt)*(X(1, :) - xA) - noneq_drift(xA, c) ).^2 );
        S = S + sum( ( (1/dt)*(xB-X(end, :)) - noneq_drift(X(end,:),c) ).^2 );
    else
        % centered difference for time: 
        %               \int |dX/dt - b(x)|^2 dt  \approx
        %               dt * \sum_i |(X_i+1 - X_i)/2*dt - b(X_i+1/2)|^2
        assert(N >= 1, ...
        "must be at least 3 discretizations including boundary for centered difference to work");
        if N == 1
        % N = 1, just dt * |(XB - XA)/2*dt - b(X1)|^2
        S = S + sum( ( (1/(2*dt))*(xB-xA) - noneq_drift(X(1,:),c) ).^2 );
        else
            % N > 1
            for i = 1:N
                % loop over each point
                if i == 1
                    % left boundary
                    S = S + sum( ( (1/(2*dt))*(X(2,:)-xA) - noneq_drift(X(i,:),c) ).^2 );
                elseif i == N
                % right boundary
                    S = S + sum( ( (1/(2*dt))*(xB-X(i-1,:)) - noneq_drift(X(i,:),c) ).^2 );
                else
                % any points in the middle of the grid
                    S = S + sum( ( (1/(2*dt))*(X(i+1,:)-X(i-1,:)) - noneq_drift(X(i,:),c) ).^2 );
                end
            end
        end
    end
    S = beta * dt * S;
end