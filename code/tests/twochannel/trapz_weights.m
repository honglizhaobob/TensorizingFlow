function weights_1d = trapz_weights(N, dx)
%       Generates trapz weights (for 1d) given number of nodes in
%       that dimension
    weights_1d = zeros(1, N)+1*dx; 
    weights_1d(1) = weights_1d(1)/2; 
    weights_1d(end) = weights_1d(end)/2;
end