function B = test_drift(x,~)
    % Manufactured valley energy potential.
    %
    % (Rescaled to [-1,1]^2)
    
    %
    % Input:
    %       x,              (N x 2) denoting the 2d points
    %
    %
    % Output:
    %       B,              (N x 2) denoting the nonequilibrium dynamics

    % create a path with 10 points, mildly quadratic in y
    xA = [-0.7,-0.4];
    xB = [0.7,-0.4];

    % create path
    path_num = 20;
    xs = linspace(xA(1),xB(1),path_num);
    ys = -0.4*ones(size(xs));
    
    % get all means for gaussian
    all_means = zeros(path_num,2);
    for i=1:path_num
        all_means(i,:)=[xs(i),ys(i)];
    end
    
    % create valley
    all_coeffs = -2*ones(path_num,1);
    
    % evaluate energy potential
    gradV = 0;
    for i = 1:path_num
        [grad_dx,grad_dy,~]=gaussian_function2b(x, all_coeffs(i),...
            all_means(i,:),8);
        grad=[grad_dx,grad_dy];
        gradV = gradV+grad;
    end
    
    
    % four Gaussians + additional penalty on mu1
    B = -gradV;
end