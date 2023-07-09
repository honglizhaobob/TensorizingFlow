function [dVdx, dVdy, V] = mixture_gaussian(x)
    % mixture Gaussian function used in Adaptive MCMC paper
    % currently used for plotting, this is also implemented
    % in energy_potential.
    
    %
    %
    % Input:
    %          x,                       (N x 2) point in 2d
    %
    % Output:
    %          V,                       (N x 1) mixture Gaussian 
    %                                   evaluated at x
    %          dVdx,                    (N x 1) derivative of V 
    %                                   w.r.t first coordinate
    %          dVdy,                    (N x 1) derivative of V 
    %                                   w.r.t second coordinate
    
    % amplitudes
    %A1 = 30; A2 = -30; A3 = -50; A4 = -50;
    A1 = 3; A2 = -5; A3 = -5; A4 = -5;
    
    % means
    mu1 = [0,1/3]; mu2 = [0,5/3]; 
    mu3 = [-1,0];  mu4 = [1,0];
    
    % compute values for Gaussian(x)
    [dg1dx, dg1dy, g1] = gaussian_function(x, A1, mu1);
    [dg2dx, dg2dy, g2] = gaussian_function(x, A2, mu2);
    [dg3dx, dg3dy, g3] = gaussian_function(x, A3, mu3);
    [dg4dx, dg4dy, g4] = gaussian_function(x, A4, mu4);
    V = g1 + g2 + g3 + g4;
    dVdx = dg1dx + dg2dx + dg3dx + dg4dx;
    dVdy = dg1dy + dg2dy + dg3dy + dg4dy;
end