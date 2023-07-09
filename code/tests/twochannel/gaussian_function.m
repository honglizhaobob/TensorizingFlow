function [dfdx, dfdy, f] = gaussian_function(x, A, mu)
    % implements a 2d Gaussian function
    % f(x, y) = A * exp( - ( (x - mu(1))^2 + (y - mu(2))^2 ) )
    % Inputs: 
    %      x,                (N x 2) vector 
    %      A,                (scalar) amplitude
    %      mu,               (2 x 1) vector, mean
    %
    %
    % Output:
    %      f,                (N x 1) f(x,y) given A and mu
    %      dfdx,             (N x 1) derivative [df/dx]
    %      dfdy              (N x 1) derivative [df/dy]
    x1 = x(:, 1); x2 = x(:, 2);
    mu1 = mu(1); mu2 = mu(2);
    f = A * exp( -( (x1 - mu1).^2 + (x2 - mu2).^2 ) );
    dfdx = f .* (-2) .* (x1 - mu1);
    dfdy = f .* (-2) .* (x2 - mu2);
end