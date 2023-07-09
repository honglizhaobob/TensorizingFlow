function p = double_rosen(theta)
    % theta is a [1,d] vector
    d = length(theta);
    sym_theta = theta;
    % use symbolic computation
    raw_func = 0;
    for k = 1:d-1
        if k <= d-3
            % rescale all dimensions by 2
            raw_func = raw_func + ((2*sym_theta(k))^2 + ...
                ((2*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        end
        if k == d-2
            % rescale (d-1) by 7, (d-2) by 2
            raw_func = raw_func + ((2*sym_theta(k))^2 + ...
                ((7*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        end
        if k == d-1
            % rescale d by 200, (d-1) by 7
            raw_func = raw_func + ((7*sym_theta(k))^2 + ...
                ((200*sym_theta(k+1)) + 5 * ((7*sym_theta(k))^2 + 1) )^2);
        end
    end

    raw_func = exp(-raw_func/2);

    raw_func2 = 0;
    % build second Rosenbrock (shifted)
    for k = 1:d-1
        if k <= d-3
            % rescale all dimensions by 2
            raw_func2 = raw_func2 + ((2*sym_theta(k))^2 + ...
                ((2*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        end
        if k == d-2
            % rescale (d-1) by 7, (d-2) by 2
            raw_func2 = raw_func2 + ((2*sym_theta(k))^2 + ...
                ((7*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        end
        if k == d-1
            % rescale d by 200, (d-1) by 7
            % flip the last dimension (d)
            raw_func2 = raw_func2 + ((7*sym_theta(k))^2 + ...
                ((200*(-sym_theta(k+1))) + 5 * ((7*sym_theta(k))^2 + 1) )^2);
        end
    end

    % sqrt of density
    raw_func2 = exp(-raw_func2/2);

    % combine the mixture
    p = log(0.5 * raw_func + 0.5 * raw_func2);
end