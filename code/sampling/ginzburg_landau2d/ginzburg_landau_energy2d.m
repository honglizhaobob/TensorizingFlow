function V = ginzburg_landau_energy2d(U, delta, d, snake)
    % Computes current system's GL energy, in 2d
    % Input:
    %       U,           (1d array) values of U1,1, U1,2, ..., U1,d
    %                                         U2,1, U2,2, ..., U2,d
    %                                         ...  ...  ... ... ...
    %                                         Ud,1, Ud,2, ..., Ud,d
    %                    then called U(:), resulting in columns of 2d U
    %                    stacked up
    
    %                    U:,0 and U:,d+1 are known to be 0.
    %       delta,       (scalar) parameter delta
    %       d,           (scalar) number of discretized nodes
    %       snake,       (binary) true, input U(:) is snake ordering
    %                             false, input U(:) is column stacked
    %
    % Output:
    %       V(U),        Ginzburg-Landau (2d) energy of the system
    
    if ~snake
        % U is column stacked up, need to rebuild by reshaping
        U = reshape(U, [d d]);
    else
        % U(:) uses snake ordering
        U = reshape(U, [d,d]);
        % even columns are flipped, flip back
        for i = 1:d
            if mod(i,2) == 0
                U(:,i) = flip(U(:,i));
            end
        end
    end
    size_U = size(U);
    d = length(U);
    assert( (size_U(1) == d) && (size_U(2) == d))
    % compute grid size h = 1/(d+1) for differencing
    h = 1/(d+1);
    % add boundary values
    U = [zeros(1,d)' U zeros(1,d)'];
    U = [zeros(1,length(U)); U; zeros(1,length(U))];
    % (u(x,y), x = 0, x = 1)
    U(1, :) = 1; U(end, :) = 1;
    % (u(x,y), y = 0, y = 1)
    U(:, 1) = -1; U(:, end) = -1;
    
    %V = 0;
    %for i = 2:(d+2)
    %    for j = 2:(d+2)
    %        V = V + ( ( U(i,j)-U(i-1,j) )/h )^2 +...
    %            ( ( U(i,j) - U(i,j-1) )/h )^2;
    %    end
    %end
    %V = (delta/2) * V;
    %V = V + ...
    %    (1/(4*delta)) * sum(sum((1-U.^2).^2));
    
    % vectorized computation
    V = (delta/2)*sum(sum( ( (1/h)*(U(2:d+2,:)-U(1:d+1,:)) ).^2 ));
    V = V + ...
       (delta/2)*sum(sum( ( (1/h)*(U(:,2:d+2)-U(:,1:d+1)) ).^2 ));
    V = V + (1/(4*delta)) * sum(sum((1-U.^2).^2));
end