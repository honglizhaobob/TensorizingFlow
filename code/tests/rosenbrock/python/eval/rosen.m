function p = rosen(theta)
    % theta is a [1,d] vector
    d = length(theta);
    sym_theta = theta;
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
    p = exp(-0.5 * raw_func);
end