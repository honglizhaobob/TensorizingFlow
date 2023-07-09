% TT Cross approximation with Ginzburg-Landau model
% set global seed
clear; clc;
rng(2, 'philox')

% 1d Ginzburg model (TT-cross Equilibrium energy)

% define parameters
delta = 0.03; 
epoch = 20; % for training data

% discretization of 1d grid for U
% U1, U2, ..., Ud
% with U0 = Ud+1 = 0 deterministic
d = 50; 

% grid size
h = 1/(d + 1); 

ginzburg_landau_energy1d(randn(1,d), delta); % make sure GLE is working

% spatial grid (1d)
x_arr = 0:h:1;


% treating each Ui as a random variable in [0,1]
% we represent the multivariate PDF using TT-cross
% evaluated on an d-dim grid
stpsz = h; % discretizing [0,1] for each dimension

gamma = 1.7; % (Yian et al. TT committor)
u_arr = -gamma:stpsz:gamma; % possible values of U, grid in each dimension

% TT meshgrid
u_tt = tt_tensor(u_arr);
U = tt_meshgrid_vert(u_tt, d);

% create discretized ginzburg-landau function
% suitable for TT cross evaluation
u_sym = sym('u', [1, d]); % symbolic U1, U2, ..., Ud
gl_func = ginzburg_landau_energy1d(u_sym, delta);
% multi-dim PDF
temp = 8; % 16, the choices come from TT committor paper (Yian et al.)
beta = 1/temp;
equi_density = exp(-beta*gl_func);
% convert to function
gl_equi_pdf_unnormalized = matlabFunction(equi_density, 'Vars', {u_sym});


% TT cross
gl_equi_pdf_unnormalized_TT = amen_cross(U, gl_equi_pdf_unnormalized, 1e-10, ...
    'y0', tt_unit(length(u_arr), d), 'nswp', 20);

disp(join(["===== sum rank of TT (unnormalized): "]))
disp(sum(gl_equi_pdf_unnormalized_TT.r))

% find normalizing constant by numerical integration (trap)
num_nodes = length(u_arr); % number of discretized nodes in each dim
weights_cell = cell(d,1);
for j = 1:d
    % fill in weights cell with compressed TT
    weights_cell{j} = tt_tensor(trapz_weights(num_nodes, stpsz));
end

% create TT meshgrid of weights
W_tt_grid = tt_meshgrid_vert(weights_cell{:});

% amen cross the weight tensor
Weights_tt = amen_cross(W_tt_grid, @(X)prod(X,2), 1e-10, 'nswp', 20);
% hadamard product F_tt * W, then sum
Z_beta = sum(gl_equi_pdf_unnormalized_TT.*Weights_tt);

% normalize to get a PDF
gl_equi_pdf_TT = gl_equi_pdf_unnormalized_TT ./ Z_beta;

% update exact symbolic functions
equi_density_func = (1/Z_beta) * exp(-beta*gl_func);


log_equi_density_func = matlabFunction(log(equi_density_func), ...
    'Vars', {u_sym});

equi_density_func = matlabFunction(equi_density_func, ...
    'Vars', {u_sym});


disp(join(["===== sum rank of TT (normalized): "]))
disp(sum(gl_equi_pdf_TT.r))

% sample from this PDF using TT-irt

% size of sample dataset
N = 2^12;

% use TT_irt_lin
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({u_arr'}, d, 1);

%[gl_TT_samples, ...
%    gl_TT_sample_densities] = tt_irt_lin(gen_grid_arr, ...
%    gl_equi_pdf_TT, unif_sample);

% use debiased TT sampler
[gl_TT_samples, ...
    gl_TT_sample_densities] = tt_irt_debias(N, log_equi_density_func, ...
    gl_equi_pdf_TT, ...
    gen_grid_arr, 'mcmc', [true]);

% get sample densities
tt_irt_sample_densities = exp(gl_TT_sample_densities);
exact_sample_densities = equi_density_func(gl_TT_samples);
% compute empirical bias
(1/N) * sum(sum(abs(tt_irt_sample_densities-exact_sample_densities)))


% scatter to check
plot_sample = false;
if plot_sample
    figure(1); grid on;
    scatter(gl_TT_samples(:,10), gl_TT_samples(:,13))
end
%plot_sample_U = [0 gl_TT_samples(randi([1,N]),:) 0];
all_energy = [];
for i = 1:length(gl_TT_samples')
    % compute individual energy
    energy_i = ginzburg_landau_energy1d(gl_TT_samples(i, :), delta);
    all_energy = [all_energy, energy_i];
end
% find 20 of the u(x) with lowest energy
[all_energy_sorted, all_energy_indices] = sort(all_energy,'ascend');
num_lowest = 10; % number of lowest energy u(x) to display
lowest_indices = all_energy_indices(1:num_lowest);
plot_samples_U = gl_TT_samples(lowest_indices,:);
% pad 0's for U0, Ud+1
plot_samples_U = [zeros(1,size(plot_samples_U,1))', plot_samples_U, ...
    zeros(1,size(plot_samples_U,1))'];



% interpolate 1d for smoothness
x_arr_fine = 0:(stpsz/10):1;
plot_samples_U_fine = zeros(size(plot_samples_U,1), length(x_arr_fine));
for i = 1:size(plot_samples_U,1)
    % interp each solution
    plot_samples_U_fine(i,:) = interp1(x_arr, plot_samples_U(i,:), x_arr_fine, ...
    'cubic');
end


verbose = true; % plot the relevant visuals

if verbose
    % plot the 1d sampled function U (randomly get one by rand int generation)
    figure(1);
    for i = 1:size(plot_samples_U, 1)
        hold on; grid on;
        plot(x_arr_fine, plot_samples_U_fine, 'LineWidth', 2);
    end
    title(join(["Sampled 1d transition state u(x) (interpolated), with avg GL energy = ", ...
        "....."]));


    %figure(2);
    %plot(x_arr, plot_sample_U, 'LineWidth', 2, 'Color', 'red'); grid on;
    %title(join(["Sampled 1d transition state u(x) (uninterpolated), with GL energy = ", ...
    %    num2str(ginzburg_landau_energy1d(plot_sample_U, delta))]));
    xlabel("x"); ylabel("u(x)");

    figure(2)
    % show the histogram of all energies
    histogram(all_energy);
end

% output data for training

% level of truncation
tolerance = 3; % 6

% truncate the PDF TT
disp(join(["===== original sum ranks = ", ...
    num2str(sum(gl_equi_pdf_TT.r))]))
gl_equi_pdf_TT_trunc = round(gl_equi_pdf_TT, tolerance);
disp(join(['===== truncated sum ranks = ', ...
    num2str(sum(gl_equi_pdf_TT_trunc.r)), " tolerance level = ", ...
    num2str(tolerance)]))

% sample from truncated TT and use as training data
%[training_data_ginsburg, ...
%    training_densities_ginsburg] = tt_irt_debias(epoch*N, ...
%    log_equi_density_func, ...
%    gl_equi_pdf_TT_trunc, ...
%    gen_grid_arr, 'mcmc', [true]);

% generate seed
unif_sample = rand(epoch*N,d);
disp(join(["===== uniform seed generated, with shape: ", ...
    num2str(size(unif_sample))]))
[training_data_ginsburg, ...
    training_densities_ginsburg] = tt_irt_lin(gen_grid_arr, ...
    gl_equi_pdf_TT_trunc, unif_sample);

% use debiased TT sampler
%[training_data_ginsburg, ...
%    training_densities_ginsburg] = tt_irt_debias(epoch*N, ...
%    log_equi_density_func, ...
%    gl_equi_pdf_TT_trunc, ...
%    gen_grid_arr, 'mcmc', [true]);


% sample a set of data from an accurate TT
% of the same size for comparison
%[comparison_data_ginsburg, ...
%    comparison_densities_ginsburg] = tt_irt_lin(gen_grid_arr, ...
%    gl_equi_pdf_TT, unif_sample);

[comparison_data_ginsburg, ...
    comparison_densities_ginsburg] = tt_irt_debias(epoch*N, ...
    log_equi_density_func, ...
    gl_equi_pdf_TT, ...
    gen_grid_arr, 'mcmc', [true]);
% 1d Ginzburg training data
filename = "../data/tt_irt_ginzburg1d_sample.mat";
save(filename)

%% 2d Ginzburg-Landau model (TT-cross Equilibrium energy)
clear; clc;
d = 10;
delta = 0.04;
temp = randn(d,d);
ginzburg_landau_energy2d(temp(:), delta, d); % make sure GLE is working
clear temp
% grid size
h = 1/(d + 1); 

% treating each Ui as a random variable in [0,1]
% we represent the multivariate PDF using TT-cross
% evaluated on an d^2-dim grid
stpsz = h;
gamma = 1.7; % size of domain to capture the solution
x_arr = 0:h:1; % spatial dimension for plotting U
u_arr = -gamma:stpsz:gamma; % grid in each dimension

% TT meshgrid
u_tt = tt_tensor(u_arr);
U = tt_meshgrid_vert(u_tt, d^2);

% create discretized ginzburg-landau function
% suitable for TT cross evaluation
u_sym = sym('u', [d, d], 'real'); % symbolic 2d U grid
gl_func = ginzburg_landau_energy2d(u_sym(:), delta, d);
% multi-dim PDF

temp = 5; % 16, the choices come from TT committor paper (Yian et al.)
beta = 1/temp;
equi_density = exp(-beta*gl_func);
% convert to function
u_sym_temp = u_sym(:);
u_sym_temp = u_sym_temp';
gl_equi_pdf_unnormalized = matlabFunction(equi_density, 'Vars', {u_sym_temp});

% TT cross
gl_equi_pdf_unnormalized_TT = amen_cross_s(U, gl_equi_pdf_unnormalized, 1e-10, ...
    'nswp', 20);

disp(join(["===== sum rank of TT (unnormalized): "]))
disp(sum(gl_equi_pdf_unnormalized_TT.r))

% find normalizing constant by numerical integration (trap)
num_nodes = length(u_arr); % number of discretized nodes in each dim
weights_cell = cell(d^2,1);
for j = 1:(d^2)
    % fill in weights cell with compressed TT
    weights_cell{j} = tt_tensor(trapz_weights(num_nodes, stpsz));
end

% create TT meshgrid of weights
W_tt_grid = tt_meshgrid_vert(weights_cell{:});

% amen cross the weight tensor
Weights_tt = amen_cross_s(W_tt_grid, @(X)prod(X,2), 1e-10, 'nswp', 20);
% hadamard product F_tt * W, then sum
Z_beta = sum(gl_equi_pdf_unnormalized_TT.*Weights_tt);

% normalize to get a PDF
gl_equi_pdf_TT = gl_equi_pdf_unnormalized_TT ./ Z_beta;

disp(join(["===== sum rank of TT (normalized): "]))
disp(sum(gl_equi_pdf_TT.r))

% sample from this PDF using TT-irt

% size of sample dataset
N = 2^14;

% generate seed
unif_sample = rand(N,d^2);
disp(join(["===== uniform seed generated, with shape: ", ...
    num2str(size(unif_sample))]))

% use TT_irt_lin
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({u_arr'}, d^2, 1);
[gl_TT_samples, ...
    gl_TT_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    gl_equi_pdf_TT, unif_sample);

% round the TT to very truncated
gl_equi_pdf_TT_rounded = round(gl_equi_pdf_TT, 1); % 3 will be rank 1
disp(join(["====== level truncated: ", ...
    num2str(sum(gl_equi_pdf_TT_rounded.r)/sum(gl_equi_pdf_TT.r))]))
% sample from the very truncated TT as training data
[training_data, training_densities] = tt_irt_lin(gen_grid_arr, ...
    gl_equi_pdf_TT_rounded, unif_sample);

% scatter to check
plot_sample = false;
if plot_sample
    figure(1); grid on;
    scatter(gl_TT_samples(:,1), gl_TT_samples(:,1+d))
end

% each sample is a 2d grid of U values, can reshape back
% and check 2d grid as following:

% check 1 sampled U function

% U function is defined on [0,1]x[0,1], with values in [-ga,ga]
[Y,X] = meshgrid(x_arr);

% compute energy of all samples

% check bimodality of training data
% delete this line if want to check
% bimodality of ground truth TT
gl_TT_samples = training_data;

gl_TT_samples_energy = zeros(N);
for i = 1:N
    U_sample_i = gl_TT_samples(i, :);
    % compute energy of sample i
    V_i = ginzburg_landau_energy2d(U_sample_i, delta, d);
    gl_TT_samples_energy(i) = V_i;
end

% find 25 minimum energy index
[gl_TT_samples_energy_sorted, gl_energy_idx] = ...
    sort(gl_TT_samples_energy,'ascend');
min_idx = gl_energy_idx(1:25); % idx for minimum energy

% show minimum energy u(x,y)
U_sample2d = reshape(gl_TT_samples(min_idx, :), [length(min_idx) d d]);
U_sample2d_copy = U_sample2d; % save for energy calculation
% pad with 0
U_sample2d = zeros(length(min_idx), d+2,d+2);
U_sample2d(:, 2:d+1, 2:d+1) = U_sample2d_copy;

% boundary conditions on unit square 
% u|x=0,1 = 1, u|y=0,1 = -1
U_sample2d(:, 1,:) = 1; U_sample2d(:, d+2,:) = 1;
U_sample2d(:, :,1) = -1; U_sample2d(:, :,d+2) = -1;

% surface plot
figure(2); 
for k = 1:25
    subplot(5, 5, k); grid on; hold on;
    surf(X, Y, squeeze(U_sample2d(k, :, :)), 'EdgeColor', 'red'); 
    shading interp; colorbar;
end

%surf(X,Y,U_sample2d,'EdgeColor','red'); shading interp; colorbar;
%title(join(["Sample 2d transition state u(x,y) (interpolated), ", ...
%    " with GL energy = ", ...
%    num2str(ginzburg_landau_energy2d(U_sample2d_copy(:), ...
%    delta, d))]))
% save data for training
filename = "../data/tt_irt_ginzburg2d_sample.mat";
save(filename)
