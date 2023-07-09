function B = noneq_drift(x, c)
    % nonequilibrium drift used in the paper
    % same as b.m
    %
    % Input:
    %       x,              (N x 2) denoting the 2d points
    %       A,              (4 x 1) scalar parameters used
    %       mu,             (4 x 2) denoting the four parameters used
    %       c,              (scalar) nonconservative coefficient
    %
    %
    % Output:
    %       B,              (N x 2) denoting the nonequilibrium dynamics
    mu1 = [0,1/3];
    mu2 = [0,5/3]; 
    mu3 = [-1,0];  
    mu4 = [1,0];
    %A1 = 30; A2 = -30; A3 = 50; A4 = 50; 
    %A1 = 30; A2 = -30; A3 = -50; A4 = -50; A5 = 0.2;
    A1 = 3; A2 = -3; A3 = -5; A4 = -5; A5 = 0.2;
    
    % drift part
    f = c * x(:, [2,1]);
    f(:,1) = -f(:,1);
    
    % get gradients from sum of Gaussian functions
    [grad1_dx, grad1_dy, ~] = gaussian_function(x, A1, mu1);
    [grad2_dx, grad2_dy, ~] = gaussian_function(x, A2, mu2);
    [grad3_dx, grad3_dy, ~] = gaussian_function(x, A3, mu3);
    [grad4_dx, grad4_dy, ~] = gaussian_function(x, A4, mu4);
    % gradient part 1
    grad1 = [grad1_dx, grad1_dy];
    grad2 = [grad2_dx, grad2_dy];
    grad3 = [grad3_dx, grad3_dy];
    grad4 = [grad4_dx, grad4_dy];
    
    % four Gaussians + additional penalty on mu1
    B = - (grad1 + grad2 + grad3 + grad4 + 4*A5*(x - mu1).^3) + f;
end