%% Test sampling on [-1,1] and [a,b] of Rosenbrock
clear; clc; rng('default');
% number of dimensions for rosenbrock
d = 11;
% create rosenbrock function
% = = = = = = = = = = 
%    cross on [a,b]
% = = = = = = = = = =
% number of nodes in each dimension
num_nodes = zeros(1,d);
for i = 1:d
    if i < d-1
        num_nodes(i) = 2^7;
    end
    if i == d-1
        num_nodes(i) = 2^9;
    end
    if i == d
        num_nodes(i) = 2^12;
    end
end

% create end points a_k for each dimension k = 1:d
% the grid is symmetrically defined on [-a_k, a_k]
end_points = zeros(1,d);
for i = 1:d
    end_points(i) = 2;
    if i == d-1
        end_points(i) = 7;
    end
    if i == d
        end_points(i) = 200;
    end
end

% create grid using TT compression
x_common = lgwt(num_nodes(1),-2,2);
x_common = x_common';
x_dm1 = lgwt(num_nodes(end-1),-7,7);
x_dm1 = x_dm1';
x_d = lgwt(num_nodes(end),-200,200);
x_d = x_d';
% store all step sizes
all_stpsz = zeros(1,d);
for i = 1:d
    all_stpsz(i) = x_common(2)-x_common(1);
    if i == d-1
        all_stpsz(i) = x_dm1(2)-x_dm1(1);
    end
    if i == d
        all_stpsz(i) = x_d(2)-x_d(1);
    end
end

x_common_tt = tt_tensor(x_common);
x_dm1_tt = tt_tensor(x_dm1);
x_d_tt = tt_tensor(x_d);

gen_grid = repmat({x_common_tt}, 1, d-2); % common grid (d-2) times
gen_grid{d-1} = x_dm1_tt; % d-1 dimension
gen_grid{d} = x_d_tt; % d dimension

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d-2, 1);
gen_grid_arr{d-1} = x_dm1';
gen_grid_arr{d} = x_d';

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);
raw_func = 0;
% use symbolic computation
for k = 1:d-1
    raw_func = raw_func + (sym_theta(k)^2 + ...
        (sym_theta(k+1) + 5 * (sym_theta(k)^2 + 1) )^2);
end
raw_func = (exp(-raw_func/2));

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
rosenbrock_tt1 = amen_cross_s(X,r_func,1e-5,'nswp',100);
%%
N = 5000;
% use TT_irt_lin
unif_sample = rand(N,d);
[rosen_TT_samples, ~] = tt_irt_lin(gen_grid_arr, ...
    rosenbrock_tt1, unif_sample);

plot_samples = true;
if plot_samples
    for i = 1:3
        figure(i); 
        if i == 1
            scatter(rosen_TT_samples(:,1), rosen_TT_samples(:,2), 5, 'blue');
            xlabel("\\theta_1"); ylabel("\\theta_2");
        end
        if i == 2
            scatter(rosen_TT_samples(:,d-2), rosen_TT_samples(:,d-1), 5, 'red');
            xlabel("\\theta_{d-1}"); ylabel("\\theta_{d}");
            xlim([-1, 0.6]); ylim([-5.5, -0.5]);
        end
        if i == 3
            scatter(rosen_TT_samples(:,d-1), rosen_TT_samples(:,d), 5, 'red');
            xlabel(sprintf("\\theta_{%d}", d-1));
            ylabel(sprintf("\\theta_{%d}", d));
            xlim([-5.5, -0.5]); ylim([-140, 0]);
        end
        title(join(["Rosenbrock Samples in Different 2d Planes,", ...
                sprintf(" d = %d", int32(d) )]));
        grid on;
    end
end
%%
clear; clc; rng("default");
% number of dimensions for rosenbrock
d = 11;
% create rosenbrock function
% = = = = = = = = = = 
%    cross on [-1,1]
% = = = = = = = = = =
% number of nodes in each dimension
num_nodes = zeros(1,d);
for i = 1:d
    if i < d-1
        num_nodes(i) = 2^7;
    end
    if i == d-1
        num_nodes(i) = 2^9;
    end
    if i == d
        num_nodes(i) = 2^12;
    end
end

% create end points a_k for each dimension k = 1:d
% the grid is symmetrically defined on [-a_k, a_k]
end_points = zeros(1,d);
for i = 1:d
    end_points(i) = 1;
    if i == d-1
        end_points(i) = 1;
    end
    if i == d
        end_points(i) = 1;
    end
end

% create grid using TT compression
x_common = lgwt(num_nodes(1),-1,1);
x_common = x_common';
x_dm1 = lgwt(num_nodes(d-1),-1,1);
x_dm1 = x_dm1';
x_d = lgwt(num_nodes(d),-1,1);
x_d = x_d';

x_common_tt = tt_tensor(x_common);
x_dm1_tt = tt_tensor(x_dm1);
x_d_tt = tt_tensor(x_d);


% final scaling
L1 = 2;
L2 = 7;
L3 = 200;

gen_grid = repmat({x_common_tt}, 1, d-2); % common grid (d-2) times
gen_grid{d-1} = x_dm1_tt; % d-1 dimension
gen_grid{d} = x_d_tt; % d dimension


% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d-2, 1);
gen_grid_arr{d-1} = x_dm1';
gen_grid_arr{d} = x_d';

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);
raw_func = 0;
% use symbolic computation
for k = 1:d-1
    if k <= d-3
        % rescale all dimensions by 2
        raw_func = raw_func + ((2*sym_theta(k))^2 + ...
            ((2*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        %raw_func = raw_func + ((200*sym_theta(k))^2 + ...
        %    ((200*sym_theta(k+1)) + 5 * ((200*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-2
        % rescale (d-1) by 7, (d-2) by 2
        raw_func = raw_func + ((2*sym_theta(k))^2 + ...
            ((7*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
        %raw_func = raw_func + ((200*sym_theta(k))^2 + ...
        %    ((200*sym_theta(k+1)) + 5 * ((200*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-1
        % rescale d by 200, (d-1) by 7
        raw_func = raw_func + ((7*sym_theta(k))^2 + ...
            ((200*sym_theta(k+1)) + 5 * ((7*sym_theta(k))^2 + 1) )^2);
        %raw_func = raw_func + ((200*sym_theta(k))^2 + ...
        %    ((200*sym_theta(k+1)) + 5 * ((200*sym_theta(k))^2 + 1) )^2);
    end
end
raw_func = (exp(-raw_func/2));

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
rosenbrock_tt2 = amen_cross_s(X,r_func,1e-5,'nswp',200);

N = 5000;
% use TT_irt_lin
unif_sample = rand(N,d);
[rosen_TT_samples, ...
    rosen_TT_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    rosenbrock_tt2, unif_sample);

for i = 1:d
    if i < d-1
        rosen_TT_samples(:,i) = L1*rosen_TT_samples(:,i);
    elseif i == d-1
        rosen_TT_samples(:,i) = L2*rosen_TT_samples(:,i);
    else
        rosen_TT_samples(:,i) = L3*rosen_TT_samples(:,i);
    end
end
%%
plot_samples = true;
if plot_samples
    for i = 1:3
        figure(i); 
        if i == 1
            scatter(rosen_TT_samples(:,1), rosen_TT_samples(:,2), 5, 'blue');
            xlabel("\\theta_1"); ylabel("\\theta_2"); 
            figure(2*i);
            histogram(rosen_TT_samples(:,1),200);
        end
        if i == 2
            scatter(rosen_TT_samples(:,d-2), rosen_TT_samples(:,d-1), 5, 'red');
            xlabel("\\theta_{d-1}"); ylabel("\\theta_{d}");
            xlim([-1, 0.6]); ylim([-5.5, -0.5]);
            figure(2*i);
        end
        if i == 3
            scatter(rosen_TT_samples(:,d-1), rosen_TT_samples(:,d), 5, 'red');
            xlabel(sprintf("\\theta_{%d}", d-1));
            ylabel(sprintf("\\theta_{%d}", d));
            xlim([-5.5, -0.5]); ylim([-140, 0]);
            figure(2*i);
        end
        title(join(["Rosenbrock Samples in Different 2d Planes,", ...
                sprintf(" d = %d", int32(d) )]));
        grid on;
    end
end

%% Test sampling from a standard Gaussian with Legendre nodes (TT-IRT)
clear; clc; rng('default');
% number of dimensions for rosenbrock
d = 2;
% create rosenbrock function
% = = = = = = = = = = 
%    cross on [a,b]
% = = = = = = = = = =
% number of nodes in each dimension
num_nodes = zeros(1,d);
for i = 1:d
    num_nodes(i) = 100;
end

% create end points a_k for each dimension k = 1:d
% the grid is symmetrically defined on [-a_k, a_k]
end_points = zeros(1,d);
for i = 1:d
    end_points(i) = 5;
end

% create grid using TT compression
x_common = lgwt(num_nodes(1),-end_points(1),end_points(1));
%x_common = linspace(-end_points(1),end_points(1),num_nodes(1));
x_common = x_common';

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);


% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% scatter for visual (Legendre nodes)
figure(1);
scatter(xq(:,1),xq(:,2),5,'blue');
grid on;

% sample using uniform nodes
figure(2);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);

% create grid using TT compression
x_common = linspace(-end_points(1),end_points(1),num_nodes(1));

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);


% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% scatter for visual (uniform spacing)
figure(3);
scatter(xq(:,1),xq(:,2),5,'red');
grid on;

figure(4);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);
%% Test sampling from 2d Gaussian, with an irregular grid (TT-IRT)
% observe if amen_cross gives segmented results
clear; clc; rng('default');

% number of dimensions for rosenbrock
d = 2;
% create rosenbrock function
% = = = = = = = = = = 
%    cross on [a,b]
% = = = = = = = = = =
% number of nodes in each dimension
num_nodes = zeros(1,d);
for i = 1:d
    num_nodes(i) = 10;
end

% create end points a_k for each dimension k = 1:d
% the grid is symmetrically defined on [-a_k, a_k]
end_points = zeros(1,d);
for i = 1:d
    end_points(i) = 5;
end

% create random irregular grid
x_common = -5+10*rand(1,num_nodes(1));

% arrange in ascending order
x_common = sort(x_common);

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);


% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% scatter for visual (Legendre nodes)
figure(1);
scatter(xq(:,1),xq(:,2),5,'blue');
grid on;

% sample using uniform nodes
figure(2);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);

% create grid using TT compression
x_common = linspace(-end_points(1),end_points(1),num_nodes(1));

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);


% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% scatter for visual (uniform spacing)
figure(3);
scatter(xq(:,1),xq(:,2),5,'red');
grid on;

figure(4);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);
%% Test sampling from 2d Gaussian, perturb sampled points (TT-IRT)
% according to Legendre spacing
clear; clc; rng("default");
d = 2;
% number of nodes in each dimension
num_nodes = zeros(1,d);
for i = 1:d
    num_nodes(i) = 100;
end

% create end points a_k for each dimension k = 1:d
% the grid is symmetrically defined on [-a_k, a_k]
end_points = zeros(1,d);
for i = 1:d
    end_points(i) = 5;
end

% create grid using TT compression
x_common = lgwt(num_nodes(1),-end_points(1),end_points(1));
%x_common = linspace(-end_points(1),end_points(1),num_nodes(1));
x_common = x_common';

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);
% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% now for each sampled coordinate, we find its 
% position in the Legendre grid, compute the 
% spacing in that subinterval, and perturb it uniformly
all_spacings = abs(x_common(2:end)-x_common(1:end-1));

% https://stackoverflow.com/questions/38958410
% /find-which-interval-a-point-b-is-located-in-matlab

% for searching, make grid in ascending order
x_common_ascend = flip(x_common);
for i = 1:Ns
    % for each sample, correct its position
    coord_x = xq(i,1);
    spacing_idx = find(x_common_ascend<coord_x,1,'last');
    spacing = all_spacings(spacing_idx);
    % perturb uniformly
    xq(i,1) = coord_x+((spacing*rand)-(spacing/2));
    
    % do the same for y
    coord_y = xq(i,2);
    spacing_idx = find(x_common_ascend<coord_y,1,'last');
    spacing = all_spacings(spacing_idx);
    % perturb uniformly
    xq(i,2) = coord_y+((spacing*rand)-(spacing/2));
end

% scatter for visual (Legendre nodes)
figure(1);
scatter(xq(:,1),xq(:,2),5,'blue');
grid on;

% sample using uniform nodes
figure(2);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);


% create grid using TT compression
x_common = linspace(-end_points(1),end_points(1),num_nodes(1));

x_common_tt = tt_tensor(x_common);

gen_grid = repmat({x_common_tt}, 1, d);

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d, 1);

% pass into tt_meshgrid to get compressed grid
X = tt_meshgrid_vert(gen_grid{:});

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);

raw_func = mvnpdf(sym_theta);


% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
gauss_tt_legendre = amen_cross_s(X,r_func,1e-1,'nswp',100);

% sample from Gaussian tensor train built on legendre nodes
Ns = 5000;
% generate uniform [0,1]^2 for seed
unif_sample = rand(Ns,d);

[xq, ~] = tt_irt_lin(gen_grid_arr, gauss_tt_legendre, unif_sample);

% scatter for visual (uniform spacing)
figure(3);
scatter(xq(:,1),xq(:,2),5,'red');
grid on;

figure(4);
subplot(1,2,1);
histogram(xq(:,1),200);
subplot(1,2,2);
histogram(xq(:,2),200);

%% Test 2d Gaussian interpolated using Legendre polynomials
clear; clc; rng("default");
d = 2;
inp = zeros(d,3);
% scaling for transformed distribution
R = 5;
% gaussian statistical profile
mu = zeros(1,2);
covmat = [1,0;0,1];

max_legendre_ord = 100;
% grid size for evaluating gaussian on
grid_size = 2000;
for legendre_ord = 41
    % number of nodes in each dimension
    num_nodes = zeros(1,d);
    for i = 1:d
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = legendre_ord*2;
    end

    % create end points a_k for each dimension k = 1:d
    % the grid is symmetrically defined on [-a_k, a_k]
    end_points = zeros(1,d);
    for i = 1:d
        end_points(i) = 1;
    end

    % create Rosenbrock function handle for d-dimensions
    sym_theta = sym('theta', [1,d]);

    raw_func = (R^2)*mvnpdf(R*sym_theta,mu,covmat);

    % convert raw function to a handle
    r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

    % compute TT-cross (used enhanced amen_cross_s from paper)
    [coeff_tt, ~] = ...
        legendre_ftt_cross(inp, r_func, legendre_ord, 1e-10);

    % get legendre data evaluated at different grid points
    grid_uniform = linspace(inp(1,1),inp(1,2),inp(1,3));
    [grid_legendre,~] = lgwt(inp(1,3),inp(1,1),inp(1,2));
    grid_legendre = grid_legendre';
    legendre_data_uniform = get_legendre(grid_uniform,legendre_ord-1,true)';
    legendre_data_legendre = get_legendre(grid_legendre,legendre_ord-1,true)';
    % contract with each legendre data and compare
    func_approx_unif = full(coeff_tt,[legendre_ord,legendre_ord]);
    func_approx_unif = tns_mult(func_approx_unif,1,legendre_data_uniform,1);
    func_approx_unif = tns_mult(func_approx_unif,1,legendre_data_uniform,1);

    func_approx_legend = full(coeff_tt,[legendre_ord,legendre_ord]);
    func_approx_legend = tns_mult(func_approx_legend,1,legendre_data_legendre,1);
    func_approx_legend = tns_mult(func_approx_legend,1,legendre_data_legendre,1);

    % exact func values on Legendre nodes
    [grid_y_legend,grid_x_legend] = meshgrid(grid_legendre);
    exact_func = matlabFunction(raw_func, 'Vars', sym_theta);
    func_exact_legend = exact_func(grid_x_legend,grid_y_legend);

    [grid_y_unif,grid_x_unif] = meshgrid(grid_uniform);
    % exact func values on uniform nodes
    func_exact_unif = exact_func(grid_x_unif,grid_y_unif);
    norm(func_exact_legend-func_approx_legend,'fro')./norm(func_exact_legend,'fro')
    norm(func_exact_unif-func_approx_unif,'fro')./norm(func_exact_unif,'fro')
    pause;
end

%% Functionally interpolate a 2d std Gaussian
clear; clc; rng("default");

d = 2;
inp = zeros(d,3);

% number of legendre polynomials
max_legendre_ord = 100;
for legendre_ord = 41
    % number of nodes in each dimension
    for i = 1:d
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = legendre_ord*2;
    end
    % create end points a_k for each dimension k = 1:d
    % the grid is symmetrically defined on [-a_k, a_k]
    end_points = zeros(1,d);
    for i = 1:d
        end_points(i) = 1;
    end

    % scaling for transformed distribution
    R = 5;

    % create Rosenbrock function handle for d-dimensions
    sym_theta = sym('theta', [1,d]);

    % mean and covariance matrix
    mu = zeros(1,2);
    covmat = [1,0;0,1];
    raw_func = (R^2)*mvnpdf(R*sym_theta,mu,covmat);

    % convert raw function to a handle
    r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

    % compute TT-cross (used enhanced amen_cross_s from paper)
    [coeff_tt, ~] = ...
        legendre_ftt_cross(inp, r_func, legendre_ord, 1e-10);

    % evaluate on denser grid and plot 2d approximate density
    % create uniform grid [-1,1]
    grid_size = 2000;
    %x_grid = linspace(-1,1,grid_size);
    %y_grid = linspace(-1,1,grid_size);
    [x_grid,~] = lgwt(grid_size,-1,1);
    x_grid = sort(x_grid)';
    
    y_grid = x_grid;
    % get legendre data on grid
    legendre_data_x = get_legendre(x_grid,legendre_ord-1,true)';
    legendre_data_y = get_legendre(y_grid,legendre_ord-1,true)';
    % contract with coefficient TT to get density 
    full_density = core(coeff_tt);
    core1 = full_density{1};
    core2 = full_density{2};
    core1 = tns_mult(core1,1,legendre_data_x,1);
    core2 = tns_mult(core2,1,legendre_data_y,1);
    % assemble into tensor train
    full_density = cell(2,1);
    full_density{1} = permute(core1,[2 1]);
    full_density{1} = reshape(full_density{1},[1 size(full_density{1})]);
    full_density{2} = core2;
    full_density = cell2core(tt_tensor(),full_density);
    full_density = full(full_density,[grid_size grid_size]);
    [y_grid_mesh,x_grid_mesh] = meshgrid(x_grid);
    figure(1);
    surf(x_grid_mesh,y_grid_mesh,full_density); 
    %plot(y_grid,full_density(1,:));
    shading interp;
    hold on;
    pause;
    % marginalized
end

%% Sample from 2d Gaussian
% building sampler
clear; clc; rng('default');
% highest number of polynomials
p = 30;
legendre_ord = p + 1; 
% generate metadata for grid
d = 3;
inp = zeros(d,3);
for i = 1:d
    inp(i,1) = -1;
    inp(i,2) = 1;
    inp(i,3) = 2^9;
end
% statistical profile
mu = zeros(1,3);
covmat = diag([1,0.5^2,0.25^2]);

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);
% scaling
R = 5;
raw_func = sqrt(mvnpdf(R*sym_theta,mu,covmat));

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% test sampling
[coeff_tt, ~] = ...
    legendre_ftt_cross(inp, r_func, legendre_ord, 1e-10);

C = coeff_tt;

% number of samples
Ns = 5000;

% create A matrix, since we built the continuous TT
% in a hyper-rectangle, we need to evaluate / sample
% from that rectangle create all grids in appropriate ranges 
% (ranges in metadata)
all_xg = cell([d,1]);
for dim = 1:d
    a_xg = inp(dim,1);
    b_xg = inp(dim,2);
    M_xg = 10000;
    % ==========
    % new code: timestamp: 11/08/2021, Hongli 
    % use legendre nodes to sample
    [xg,~] = lgwt(M_xg,-1,1);
    % set grid to be ascending order
    xg = flip(xg)'; % need to be a row vector
    % nonuniform spacing
    all_spacings = abs(xg(2:end)-xg(1:end-1));
    xg = xg + ([all_spacings,0]./2); % the 0 is dummy, we don't use it
    xg(end) = [];
    % ==========
    % ==========
    % old code: timestamp: 11/08/2021, Hongli 
    % use uniform nodes to sample
    %xg = linspace(-1,1,M_xg); % first create [-1,1] 
    %dx = xg(2)-xg(1);
    %xg = xg+dx/2;
    %xg(end) = [];
    % ==========
    
    
    
    % sample from [-1,1] and rescale samples in each dimension
    all_xg{dim} = xg;
end
% build all legendre data (with shifted domain)
all_A = cell([d,1]);
for dim = 1:d
    xg = all_xg{dim};
    all_A{dim} = get_legendre(xg,p,true)';
end

% truncate the coefficient train
max_rank = 5;
C = round(C, 1e-10, max_rank);
C = C./norm(C); % renormalize after rounding, since rounding breaks norm

% continuous tensor train samples
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    X(:,s) = get_sample(C,all_A,all_xg);
end

%% 
% visualize samples
figure(1);
scatter(R*X(1,:), R*X(2,:), 5, 'blue');
grid on;
figure(2);
scatter(R*X(2,:), R*X(3,:), 5, 'red');
grid on;

% sample using MATLAB
X_matlab = mvnrnd(mu,covmat,Ns)';

figure(3);
scatter(X_matlab(1,:), X_matlab(2,:), 5, 'blue');
grid on;
figure(4);
scatter(X_matlab(2,:), X_matlab(3,:), 5, 'red');
grid on;

%% Sampling from a scaled Gaussian with nonuniform domain
clear; clc; rng('default');

% highest number of polynomials
p = 30;
legendre_ord = p + 1; 
% generate metadata for grid
% d-dimensional Gaussian with gradually increasing variance
d = 9;
mu = zeros([1,d]);
sigma = randn(9,9);
for i = 1:9
    sigma(i,i) = i;
end
sigma = sigma'*sigma;
% sample size
N = 5000;
% analytic samples
X_matlab = mvnrnd(mu,sigma,N);
% visualize last two dimensions
figure(1);
for i = 1:8
    subplot(4,2,i);
    scatter(X_matlab(:,i),X_matlab(:,i+1));
    hold on;
end
%%
% set up domain for continuous sampling
inp = zeros(d,3);
for i = 1:d
    if i < d-1
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^8;
    end
    if i == d-1
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^8;
    end
    if i == d
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^8;
    end      
end

% final scaling 
L1 = 10;
L2 = 15;
L3 = 20;

% compute grid step sizes
grid1 = linspace(inp(1,1),inp(1,2),inp(1,3));
grid2 = linspace(inp(d-1,1),inp(d-1,2),inp(d-1,3));
grid3 = linspace(inp(d,1),inp(d,2),inp(d,3));
dx1 = grid1(2)-grid1(1);
dx2 = grid2(2)-grid2(1);
dx3 = grid3(2)-grid3(1);

% create Gaussian function handle for d-dimensions
sym_theta2 = sym('theta', [1,d]);
sym_theta = sym_theta2;
for i = 1:d
    if i < d-1
        sym_theta(i) = L1*sym_theta(i);
    elseif i == d-1
        sym_theta(i) = L2*sym_theta(i);
    else
        sym_theta(i) = L3*sym_theta(i);
    end
end

% function handle
gauss_sqrt_pdf = sqrt(mvnpdf(sym_theta,mu,sigma));

% convert raw function to a handle
r_func = matlabFunction(gauss_sqrt_pdf, 'Vars', {sym_theta2});

% test sampling
[coeff_tt, basisq] = ...
    legendre_ftt_cross(inp, r_func, legendre_ord, 1e-10);


%%
C = coeff_tt;
% number of samples
Ns = 5000;

% create A matrix, since we built the continuous TT
% in a hyper-rectangle, we need to evaluate / sample
% from that rectangle create all grids in appropriate ranges 
% (ranges in metadata)
all_xg = cell([d,1]);
for dim = 1:d
    a_xg = inp(dim,1);
    b_xg = inp(dim,2);
    M_xg = 10000;
    xg = linspace(-1,1,M_xg); % first create [-1,1] 
    dx = xg(2)-xg(1);
    xg = xg+dx/2;
    xg(end) = [];
    % sample from [-1,1] and rescale samples in each dimension
    all_xg{dim} = xg;
end
% build all legendre data (with shifted domain)
all_A = cell([d,1]);
for dim = 1:d
    xg = all_xg{dim};
    all_A{dim} = get_legendre(xg,p,true)';
end
% truncate the coefficient train
%max_rank = 5;
%C = round(C, 1e-10, max_rank);
C = C./norm(C); % normalize after rounding, since rounding breaks norm

% continuous tensor train samples
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    X(:,s) = get_sample(C,all_A,all_xg);
end

% rescale all samples in each dimension
new_X = X;
for dim = 1:d
    if dim <= d-2
        new_X(dim,:) = L1.*new_X(dim,:);
    elseif dim == d-1
        new_X(dim,:) = L2.*new_X(dim,:);
    else
        new_X(dim,:) = L3.*new_X(dim,:);
    end
end

% visualize last two dimensions
figure(3); 
scatter(new_X(d-2,:),new_X(d-1,:));
figure(4);
scatter(new_X(d-1,:),new_X(d,:));
