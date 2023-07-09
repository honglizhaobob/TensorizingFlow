function V = ginzburg_landau_energy1d(U, delta)
    % Computes current system's GL energy, in 1d
    % Input:
    %       U,           (1d array) values of U1, U2, ..., Ud
    %                    U0 and Ud+1 are known to be 0.
    %       delta,       (scalar) parameter delta
    %
    % Output:
    %       V(U),        Ginzburg-Landau (1d) energy of the system
    
    d = length(U);
    % default make row vector
    U = reshape(U, [1 length(U)]);
    % add U0 and Ud+1
    %U = [0, U, 0];
    
    % compute grid size h = 1/(d+1) for differencing
    h = 1/(d+1);
    % compute energy
    V = sum((delta/2).*( ((1/h) * (U(2:end) - U(1:end-1))).^2 ) + ...
        (1/(4*delta)) .* ( (1-U(2:end).^2).^2 ));
end