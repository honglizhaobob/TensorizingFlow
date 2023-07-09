% Follow example in Adaptive Monte Carlo augmented with normalizing Flows
% nonequilibrium path sampling using TT

% set global seed
clear; clc; 
format long
rng("default")

% parameter choices from [Gabrie]
beta = 1/20000;
t_start = 0; t_end = 0.6;
N = 10; % tune N for finer solutions, effective dimension 2N
dt = (t_end-t_start)/N;
temperature = 1/beta;
dt = 0.5;

% boundary conditions
xA = [-1; 0]; xB = [1; 0]; 

% need to discretize 2d grid for the path
L = 2;  k = 50;

% for a single point in 2D, there are N points
X_arr = linspace(-1.2,1.2,k);
Y_arr = linspace(-0.7,2,k);

% compress MATLAB array
X_tt = tt_tensor(X_arr);
Y_tt = tt_tensor(Y_arr);

% TT meshgrid
%X = tt_meshgrid_vert(X_tt, N*2); % effective dimension 2N
% TT meshgrid, variable ordering (x1,x2,...,xN,y1,y2,...,yN)
tmp = cell(1,2*N);
for i = 1:N
    tmp{i} = X_tt;
end

for i = N+1:2*N
    tmp{i} = Y_tt;
end
X = tt_meshgrid_vert(tmp(:));

% get symbolic function
U = sym('u', [N, 2], 'real');
path_action = energy_potential(U(:), dt, beta);

% convert to function
u_sym_temp = U(:);
u_sym_temp = u_sym_temp';

path_pdf = exp(-0.25 * path_action);


% function handle for PDF (formal)
path_pdf_func = matlabFunction(path_pdf, ...
    'Vars', {u_sym_temp});

% log function handle, the energy U
log_path_pdf_func = matlabFunction(-0.25 * path_action, ...
    'Vars', ...
    {u_sym_temp});

% TT cross PDF (formal)
path_pdf_func_tt = amen_cross_s(X, path_pdf_func, 1e-10, 'nswp', 15);
sum_rank_before_trunc = sum(path_pdf_func_tt.r);
% severely truncate the TT
path_pdf_func_tt2 = round(path_pdf_func_tt, 3);

disp(strcat("> Percentage of TT Truncation = ", ...
    num2str(sum(path_pdf_func_tt.r)/sum_rank_before_trunc)));
disp(strcat("> Range of PDF tensor = ", ...
    num2str(tt_min(path_pdf_func_tt)), ...
    ", ", num2str(tt_max(path_pdf_func_tt))))

% generate uniform seed
M = 2^10; % number of sample points
unif_sample = rand(M,2*N);
disp(join(["===== uniform seed generated, with shape: ", ...
    num2str(size(unif_sample))]))
% use TT_irt_lin
% create cell array representing grid (MATLAB arrays)
%gen_grid_arr = repmat({X_arr'}, 2*N, 1);
% truncated domain
gen_grid_arr = cell(2*N,1);
for i = 1:N
    gen_grid_arr{i} = X_arr';
end
for i = N+1:2*N
    gen_grid_arr{i} = Y_arr';
end
[path_samples, path_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    path_pdf_func_tt, unif_sample);
%[path_samples, ...
%    path_sample_densities] = tt_irt_debias(M, log_path_pdf_func, ...
%    path_pdf_func_tt, ...
%    gen_grid_arr, 'mcmc', [true]);

% plot a few path samples
% reshape into (M x 2*N)
plot_path_samples = reshape(path_samples, [M N 2]);

for i = 1:M
    path_i = squeeze(plot_path_samples(i, :, :));
    % add boundary
    path_i = [xA.'; path_i; xB.'];
    figure(1); hold on;
    plot(path_i(:, 1), path_i(:, 2), 'LineWidth', 1.2)
    xlim([-2 2]); ylim([-2 2]);
    hold on;
end

% evaluate the energy of each sample
all_energy = zeros(1, M);
for i = 1:M
    sample_path = squeeze(plot_path_samples(i,:));
    all_energy(i) = energy_potential(sample_path, ...
        dt, beta);
end
disp(strcat("= range of energy:  ", ...
    num2str([min(all_energy), max(all_energy)])))
% save data
filename = "../data/tt_irt_path_sampling.mat";
save(filename)

%% Nonequilibrium Sampling with SVD Tensor Train
% set global seed
clear; clc; 
format long
rng("default")

% parameters (need to fine tune beta and dt)
beta = 1;
t_start = 0; t_end = 0.6;
N = 10; % tune N for finer solutions, effective dimension 2N
dt = (t_end-t_start)/N;
temperature = 1/beta;
dt = 0.1;

% boundary conditions
xA = [-1; 0]; xB = [1; 0]; 

% need to discretize 2d grid for the path
L = 2;  k = 21;

% assume uniform grid
X_arr = linspace(-L,L,k);

% compress MATLAB array
X_tt = tt_tensor(X_arr);

% TT meshgrid
%X = tt_meshgrid_vert(X_tt, N*2); % effective dimension 2N
% TT meshgrid, variable ordering (x1,x2,...,xN,y1,y2,...,yN)
X = tt_meshgrid_vert(X_tt,2*N);

% SVD TT PDF (formal)
svd_path_pdf_tt = approx_transition_kernel(X_arr, N, 1e-15, dt, beta);

% generate uniform seed
M = 2^5; % number of sample points
unif_sample = rand(M,2*N);
disp(join(["===== uniform seed generated, with shape: ", ...
    num2str(size(unif_sample))]))
% use TT_irt_lin
% create cell array representing grid (MATLAB arrays)
% truncated domain
gen_grid_arr = cell(2*N,1);
for i = 1:2*N
    gen_grid_arr{i} = X_arr';
end

[path_samples, path_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    svd_path_pdf_tt, unif_sample);

% plot a few path samples
% reshape into (M x 2*N)
% ordering: xN_2, xN_1, xN-1_2, xN-1_1, ..., x1_2, x1_1, x0_2, x0_1

plot_path_samples = zeros(M,N,2);
for i = 1:M
    i
    sample_path = path_samples(i,:);
    plot_path_samples(i,:,:) = reshape(flip(sample_path), [2 N]).';
end

for i = 1:M
    path_i = squeeze(plot_path_samples(i, :, :));
    % add boundary
    path_i = [xA.'; path_i; xB.'];
    figure(1); hold on;
    plot(path_i(:, 1), path_i(:, 2), 'LineWidth', 1.2)
    xlim([-2 2]); ylim([-2 2]);
    hold on;
end

% evaluate the energy of each sample
all_energy = zeros(1, M);
for i = 1:M
    sample_path = squeeze(plot_path_samples(i,:));
    all_energy(i) = energy_potential(sample_path, ...
        dt, beta);
end
disp(strcat("= range of energy:  ", ...
    num2str([min(all_energy), max(all_energy)])))
%% Nonequilibrium Sampling with MALA
% NOTE: this section implements Langevin Monte Carlo 
% sampling and is independent of the previous section
% set global seed
clear; clc; 
format long
rng("default")

% parameters
beta = 4;
temperature = 1/beta;
t_start = 0; t_end = 0.6;

dtau = 5e-5;
actual_N = 102; % variables + 2 fixed boundary points
N = actual_N - 2;
dt = 6e-3;
% boundary conditions
xA = [-1; 0]; xB = [1; 0]; 

% create gradient function
gradient_S_func = grad_energy_potential_sym(N, dt, beta);

% loop for Langevin MC sampling
k_max = 5000; % number of sampling steps
k = 1;
metropolize = true; % - true: MALA
                     % - false: ULA

% take canonical path as initial data
init_data = zeros(actual_N, 2); 

upper_initialization = true;
if upper_initialization
    % upper channel init
    % take a spline from (-1,0)->(0,5/3), (0,5/3)->(1,0) as initial data
    init_data(1:round(actual_N/2), :) = [linspace(-1, 0, round(actual_N/2)).', ...
    ((3/2)*linspace(-1, 0, round(actual_N/2)) + 3/2).'];
init_data(round(actual_N/2)+1:actual_N, :) = [linspace(0, 1, actual_N-round(actual_N/2)).', ...
    ((-3/2)*linspace(0, 1, actual_N-round(actual_N/2)) + 3/2).'];
else
    % lower channel init
    init_data(:, 1) = linspace(-1, 1, actual_N);
end

% only take x1, x2, ..., xN-1
init_data = init_data(2:end-1, :);

all_data = zeros(k_max, 2*N); % store all sampled paths (w/o boundary)
curr_path = init_data;
prev_path = 0;
num_rejected = 0; % number of rejected samples by metropolis step
while k <= k_max
    k
    % compute gradient
    curr_path_arg = num2cell(curr_path(:));
    % gradient has shape Nx2
    grad_S = gradient_S_func(curr_path_arg{:});
    % Langevin update each point in the path
    prev_path = curr_path;
    curr_path = curr_path - dtau * grad_S;
    % add dWt/dt
    eta = reshape(mvnrnd(zeros(1,2*N), eye(2*N)), [N 2]);
    curr_path = curr_path + sqrt(2 * dtau) .* eta;
    % metropolize if needed
    if metropolize
        % compute PDF values at paths 
        log_pdf_curr = -0.25 * energy_potential(curr_path(:), dt, beta);
        log_pdf_prev = -0.25 * energy_potential(prev_path(:), dt, beta);
        % compute transition prob
        log_q_prev_to_curr = log(transition_prob(prev_path, curr_path,...
            grad_S, dtau, true)); % forward: X_k --> X_k+1 = q(X_k+1|X_k)
        log_q_curr_to_prev = log(transition_prob(prev_path, curr_path,...
            grad_S, dtau, false)); % backward: X_k+1 --> X_k = q(X_k|X_k+1)
        % compute acceptance 
        acceptance = exp((log_pdf_curr + log_q_curr_to_prev) - ...
            (log_pdf_prev + log_q_prev_to_curr));
        
        alpha = min(1, acceptance);
        % sample uniform and check U(0,1) <? alpha
        u = rand();
        if u <= alpha
            % accept and record the path
            all_data(k, :) = curr_path(:);
        else
            % reject 
            num_rejected = num_rejected + 1;
            if mod(num_rejected, 200) == 0
                disp(["=== Rejected, total number rejected = ", ...
                    num2str(num_rejected)]);
            end
            curr_path = prev_path;
        end
    else
        % record the path
        all_data(k, :) = curr_path(:);
    end
    % next sampling iter
    k = k + 1;
end

% get rid of zero rows in all_data
zero_row_ind = find(sum(all_data, 2) == 0);
all_data_clean = all_data;
all_data_clean(zero_row_ind, :) = [];

figure(2)
% plot samples from ULA
for i = 1:length(all_data_clean)
    sample_path = reshape(all_data_clean(i, :), [N 2]);
    % pad sample path with end points
    sample_path = [xA.'; sample_path; xB.'];
    plot(sample_path(:, 1), sample_path(:, 2)); hold on; grid on;
    xlim([-1.5, 1.5]); ylim([-1.5, 1.5]);
    if metropolize
        title("Samples from MALA");
    else
        title("Samples from Unadjusted Langevin MC");
    end
end

all_energy = zeros([1, length(all_data_clean)]);
for i = 1:length(all_data_clean)
    sample_path = reshape(all_data_clean(i, :), [N 2]);
    % boundary is penalized in ENERGY_POTENTIAL
    all_energy(i) = energy_potential(sample_path(:), dt, beta);
end

% plot histogram of energy potentials for sampled paths
figure(3); hold on;
title_string = strcat("Histogram of Energy: ", num2str(k_max), ...
    " samples, ");
if upper_initialization
    title_string = strcat(title_string, "Upper Channel Initialization");
else
    title_string = strcat(title_string, "Lower Channel Initialization");
end
title(title_string);
histogram(all_energy); xlabel("Energy Level"); ylabel("Density");
%% Build Tensor Train with SVD and check PDF values
% check SVD values
addpath("./SVD-TT-Toolbox");
% parameters (need to fine tune beta and dt)
t_start = 0; t_end = 0.6;
N = 100; % tune N for finer solutions, effective dimension 2N
dt = (t_end-t_start)/N;
temperature = 1/beta;
dt = 6e-3;

% boundary conditions
xA = [-1; 0]; xB = [1; 0]; 

% need to discretize 2d grid for the path
L = 2;  k = 29;

% assume uniform grid
X_arr = linspace(-L,L,k);

% SVD TT PDF (formal)
svd_path_pdf_tt = approx_transition_kernel(X_arr, N, 1e-10, dt, beta);
norm(svd_path_pdf_tt)
% search indices of sampled paths in the TT, for indexing
svd_pdf = zeros(length(all_data_clean), 1);
for i = 1:length(all_data_clean)
    % get path
    sample_path = all_data_clean(i,:);
    % reshape to match ordering of SVD TT
    sample_path = reshape(flip(sample_path(:)), [N 2])';
    % find nearest indices
    svd_tt_ind = dsearchn(X_arr', sample_path(:));
    svd_pdf(i) = svd_path_pdf_tt(svd_tt_ind);
end
%% helper functions
function q = transition_prob(X_prev, X_curr, gradS_prev, dtau, forward)
    % Implements q(X_i+1|X_i) the transition probability 
    % for Langevin MC
    %
    %   Inputs:
    %       X_prev,                 previous Langevin MC step (Nx2)
    %       X_curr,                 current Langevin MC step to be
    %                               accepted (Nx2)
    %       gradS,                  (N x 2) energy gradient evaluated 
    %                               at X_prev
    %       forward,                (bool)   if true, computes q(i+1|i)
    %                                        if false, computes q(i|i+1)
    %       dtau,                   time step size used by chain
    %
    %
    %   Outputs:
    %       q,                      desired transition probability
    N = round(size(X_curr, 1));
    if forward
        % foward transition prob
        q = mvnpdf(X_curr(:), X_prev(:) - dtau.* gradS_prev(:), ...
            2*dtau.*eye(2*N));
    else
        % backward transition prob
        q = mvnpdf(X_prev(:), X_curr(:) + dtau.* gradS_prev(:), ...
            2*dtau.*eye(2*N));
    end
end