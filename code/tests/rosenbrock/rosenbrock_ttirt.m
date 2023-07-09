%% experiments with TT sampling Gaussian
% code found in https://github.com/dolgov/TT-IRT

clear
% reproducibility
rng("default");

% test a 2d Gaussian with rectangular grid
clear

n1 = 1000; n2 = 1000;
x = linspace(-5,5,n1)';        % 1d grids
y = linspace(-5,5,n2)';
dx = x(2)-x(1); dy = y(2)-y(1);
xtt = tt_tensor(x);            % Convert to TT
ytt = tt_tensor(y);


% 2d meshgrid in TT
X = tt_meshgrid_vert(xtt,ytt);  

% standard Gaussian in 2d
fun = @(Y) (1/(2*pi))*exp(-0.5*(Y(:,1).^2 + Y(:,2).^2));
log_fun = @(Y) log(fun(Y));

%%
% Output function as 2-dimensional TT format 
ftt = amen_cross_s(X,fun,1e-10,'nswp',200, 'verb', 0);

plot = false;
if plot
    surf(full(X{1}, [n1 n2]), full(X{2}, [n1 n2]), full(ftt, [n1 n2])); 
    shading interp;
end

% use pre-built sampler to sample
M = 2000; 
% create cell array representing grid
xsf = {x;y};
% generate uniform [0,1]^2 for seed
unif_sample = rand(M,length(xsf));
% use TT_irt_lin
%[xq, lFapp] = tt_irt_lin(xsf, ftt, unif_sample);
[xq, lFapp] = tt_irt_debias(M, log_fun, ftt, xsf, 'mcmc', [true])
% actual sample from 2d Gaussian
xq_exact = mvnrnd(0*ones(2,1), eye(2), M);

% show sample
show_sample = false;
if show_sample
    scatter(xq(:,1), xq(:,2), 12, 'blue'); hold on;
    scatter(xq_exact(:,1), xq_exact(:,2), 12, 'red');
end

% test correlated Gaussian
sigma = randn(2,2); sigma = sigma'*sigma;
inv_sigma = inv(sigma);
fun2 = @(X) (1/(2*pi)) * (det(sigma)^(-1/2)) * ...
    exp(-0.5 * (inv_sigma(1,1)*X(:,1).^2 + (inv_sigma(1,2)+inv_sigma(2,1))* ...
    X(:,1).*X(:,2) + inv_sigma(2,2)*X(:,2).^2));

ftt = amen_cross_s(X,fun2,1e-10,'nswp',20,'trunc_method','svd', 'verb', 0);
% look at distribution
%plot = false;
%if plot
%    surf(full(X{1}, [n1 n2]), full(X{2}, [n1 n2]), full(ftt, [n1 n2])); 
%    shading interp;
%end

% use pre-built sampler to sample
M = 2000; 
% create cell array representing grid
xsf = {x;y};
% generate uniform [0,1]^2 for seed
unif_sample = rand(M,length(xsf));
% use TT_irt_lin
[xq, lFapp] = tt_irt_lin(xsf, ftt, unif_sample);

% actual sample from 2d Gaussian
xq_exact = mvnrnd(0*ones(2,1), sigma, M);

% show sample
show_sample1 = true;
show_sample = false;
if show_sample1
    scatter(xq(:,1), xq(:,2), 12, 'blue'); hold on;
    scatter(xq_exact(:,1), xq_exact(:,2), 12, 'red');
end

% rank truncation
ftt_rounded = round(ftt, 1e-14, [1,16,1]);
% print error
disp(">>> displaying truncation error")
disp(norm(ftt - ftt_rounded))

% use the severely truncated TT to generate samples and compare
% with actual gaussian

% use pre-built sampler to sample
M = 1000; 

% show sample
show_sample = false;
% generate uniform [0,1]^2 for seed
unif_sample = rand(M,length(xsf));
if show_sample
% save a sequence of rank truncations
figure(1); sgtitle(join(['TT sampling of correlated Gaussian with',...
    ' increasingly severe rank truncation'])); 
for i = 1:16
    subplot(4,4,i);
    % create cell array representing grid
    xsf = {x;y};
    % round the TT prob distribution
    ftt_rounded = round(ftt, 1e-14, [1,16+1-i,1]);
    % use TT_irt_lin
    [xq, lFapp] = tt_irt_lin(xsf, ftt_rounded, unif_sample);

    % actual sample from 2d Gaussian
    xq_exact = mvnrnd(0*ones(2,1), sigma, M);
    scatter(xq(:,1), xq(:,2), 12, 'blue'); 
    xlabel("$x_1$"); 
    ylabel("$x_2$"); 
    hold on;
    %scatter(xq_exact(:,1), xq_exact(:,2), 12, 'red');
    ranks = num2str(ftt_rounded.r');
    % compute error to initial distribution in Frobenius
    fro_error = norm(ftt_rounded - ftt);
    title(join(["rank: ", ranks, "error = ", fro_error]))
    grid on;
end

end
%% Test sampling from High Dimensional (d>2) Gaussian and project
% in the end, project onto (x1,x2) for visualization
rng("default"); % for reproducibility
% exact high-d correlated Gaussian
d = 11;
mu = 2*randn(d,1);
sigma = randn(d,d); sigma = sigma'*sigma;
% generate samples
N = 6000;
s_exact = mvnrnd(mu, sigma, N);

% TT truncated sampling
% generate seed first in [0,1]^d
unif_sample = rand(N,d);

x_start = -20; x_end = 20;
x = linspace(x_start, x_end, N); x = x';
% compress
xtt = tt_tensor(x);
% generate meshgrid
X = tt_meshgrid_vert(xtt, d);
inv_sigma = inv(sigma);

% use symbolic package to compute the Gaussian PDF function handle
sym_x = sym("x", [1,d]);
sym_x_t = sym_x';
gaussian_pdf = ((2*pi)^(-d/2)) * (det(sigma)^(-0.5)) * ...
    exp(-0.5*(sym_x-mu')*inv_sigma*(sym_x-mu')');
% create function handle
func = matlabFunction(gaussian_pdf, 'Vars', {sym_x});

% use TT_cross to interpolate the PDF
ftt = amen_cross(X,func,1e-10,'nswp',20);

% rank truncation
ftt_rank_before = ftt.r;
trunc_tol = 3;
ftt = round(ftt, trunc_tol);
% ftt = round(ftt, trunc_tol, [1,3,5,3,1])

% start sampling
% create cell array representing grid
xsf = repmat({x},d,1);
% generate uniform [0,1]^2 for seed
unif_sample = rand(N,length(xsf));
% use TT_irt_lin
[xq, lFapp] = tt_irt_lin(xsf, ftt, unif_sample);

% compare two samples
plot_samples = true;
if plot_samples
    % plot samples projected on (x1, x2)
    figure(1); sgtitle(join(['TT sampling of correlated Gaussian Heatmap, ', ...
        'sample size = ', num2str(N)])); 
    plotnum = 0;
    for i = 1:d
        for j = 1:d
            plotnum = plotnum + 1;
            subplot(d,d,plotnum); 
            scatter(s_exact(:,j), s_exact(:,i), 12, 'red'); hold on;
            scatter(xq(:,j), xq(:,i), 12, 'blue'); grid on;
            xlabel(i);
            ylabel(j);
        end
    end
end

ftt_rank_after = ftt.r;
%% Normalizing Flow Correction

% output the sample points generated from 
% a high-d Gaussian (severely truncated)
% we use Python to construct NF

filename = 'tt_irt_sample.mat';
save(filename);
%% Test sampling from distribution induced by Rosenbrock function part I
% building sampler
clear; clc;
rng(35, "philox")
% number of dimensions
d = 11;

% define grid according to paper: Dolgov et al.

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
x_common = linspace(-end_points(1), end_points(1), num_nodes(1));
x_dm1 = linspace(-end_points(d-1), end_points(d-1), num_nodes(d-1));
x_d = linspace(-end_points(d), end_points(d), num_nodes(d));

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
raw_func = exp(-raw_func/2);

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% compute TT-cross (used enhanced amen_cross_s from paper)
rosenbrock_tt = amen_cross_s(X,r_func,1e-10,'nswp',200);

% we need to find the normalizing factor by numerical integration
% of Rosenbrock TT as a multivariate function
weights_cell = cell(length(num_nodes),1);
% use TT integration to find normalizing constant
for j = 1:length(num_nodes)
    % generate a cell matrix of weights
    % weight vectors are row vectors
    % the matrix should be of dimension
    % d x N_d (N_d is number of function nodes in
    % that dimension).
    N_nodes = num_nodes(j);
    % compress to TT
    weights_cell{j} = tt_tensor(trapz_weights(N_nodes, all_stpsz(j)));
end

% create TT meshgrid of weights
W_tt_grid = tt_meshgrid_vert(weights_cell{:});
% amen cross the weight tensor
Weights_tt = amen_cross(W_tt_grid, @(X)prod(X,2), 1e-12, 'nswp', 20);
% hadamard product F_tt * W, then sum
rosen_norm_const = sum(rosenbrock_tt.*Weights_tt);
% normalize the rosenbrock
rosenbrock_tt = rosenbrock_tt ./ rosen_norm_const;

% update r_func by the normalizing constant
r_func = matlabFunction(raw_func/rosen_norm_const, 'Vars', {sym_theta});
log_r_func = matlabFunction(log(raw_func/rosen_norm_const),...
    'Vars', {sym_theta});

disp(join([">>> integral of normalized", ...
    num2str(sum(rosenbrock_tt.*Weights_tt))]))

% start sampling
% create cell array representing grid (MATLAB arrays)
gen_grid_arr = repmat({x_common'}, d-2, 1);
gen_grid_arr{d-1} = x_dm1';
gen_grid_arr{d} = x_d';

% number of samples
N = 5000; % 2^17

% generate uniform [0,1]^d for seed

% pre-generate all samples for training purpose
epoch = 100; % 100
unif_sample = rand(N,d);
disp(join([">>> uniform seed generated, with shape: ", ...
    num2str(size(unif_sample))]))

% use TT_irt_lin
[rosen_TT_samples, ...
    rosen_TT_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    rosenbrock_tt, unif_sample);
%[rosen_TT_samples, ...
%    rosen_TT_sample_densities] = tt_irt_debias(N, log_r_func, ...
%    rosenbrock_tt, ...
%    gen_grid_arr, 'mcmc', [true]);


% the TT-IRT alg. outputs approximate (log) densities at sampled points
% we can verify:
tt_irt_sample_app_densities = exp(rosen_TT_sample_densities);
% evaluate density via exact function
exact_sample_densities = r_func(rosen_TT_samples);
% compute average difference
(1/(epoch*N)) * (sum(sum(abs(exact_sample_densities-tt_irt_sample_app_densities))))

idx = randi([1, size(tt_irt_sample_app_densities, 1)], N);
% create overlaid histogram
show_hist = false; % show histogram
if show_hist
    figure(1);
    [counts1, binCenters1] = hist(tt_irt_sample_app_densities, 500);
    [counts2, binCenters2] = hist(exact_sample_densities, 500);
    plot(binCenters1, counts1, 'r', 'LineWidth', 1.5);
    hold on;
    plot(binCenters2, counts2, 'b', 'LineWidth', 1.5); grid on;
    % Put up legend.
    legend1 = sprintf('TT Sampled Densities');
    legend2 = sprintf('Exact Densities');
    legend({legend1, legend2});
end

% display size of samples
disp(join([">>> TT successfully sampled, sample size = ", ...
    num2str(size(rosen_TT_samples))]))

% similar to the original paper, should have heavy tail in the (d-1,d)
% plane

% scatter (theta1, theta2), (theta2, theta3), (theta{d-1}, theta{d})
% Rosenbrock sampling part II: plot samples
plot_samples = true;
if plot_samples
    for i = 1:3
        figure(i); 
        if i == 1
            scatter(rosen_TT_samples(:,1), rosen_TT_samples(:,2), 5, 'blue');
            xlabel("\theta_1"); ylabel("\theta_2");
        end
        if i == 2
            scatter(rosen_TT_samples(:,d-2), rosen_TT_samples(:,d-1), 5, 'red');
            xlabel("\theta_{d-1}"); ylabel("\theta_{d}");
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
% Rosenbrock sampler part III: severe rank truncation and Wasserstein metric
% accuracies used for rank truncation
ftt = rosenbrock_tt;
round_accs = [1./10.^(0:10), 2, 3];
round_accs = flip(round_accs); % empirical tested, change later for rigor
                                    % 3 should eliminate all meaningful
                                    % ranks


truncate = true;
if truncate
% save a sequence of rank truncations
figure(1); sgtitle(join(['TT sampling of Rosenbrock with',...
    ' increasingly severe rank truncation'])); 
% use a randomly sampled subset of data to test accuracy
% because unif_sample = (epoch x N) x d
resampled = datasample(unif_sample, N);
for i = 1:length(round_accs)
    subplot(4,4,i);
    % check cell array, unif seeds exist from Part I
    gen_grid_arr; unif_sample;
    % round the TT prob distribution
    ftt_rounded = round(ftt, round_accs(i));
    % use TT_irt_lin
    %[xq, lFapp] = tt_irt_lin(gen_grid_arr, ftt_rounded, ...
    %    resampled);
    
    % use debiased TT_irt
    [xq, lFapp] = tt_irt_debias(N, log_r_func, ftt_rounded, ...
        gen_grid_arr, 'mcmc', [true]);
    
    % check untruncated sample exists from Part I
    rosen_TT_samples; d;
    % sample from untruncated TT from Part I
    %[xq_accurate, lFapp_accurate] = tt_irt_lin(gen_grid_arr, ftt, ...
    %    resampled);
    
    % use debiased TT_irt
    [xq_accurate, lFapp_accurate] = tt_irt_debias(N, log_r_func, ftt, ...
        gen_grid_arr, 'mcmc', [true]);
    
    % specifically plot the heavy tail
    scatter(xq(:,d-1), xq(:,d), 5, 'blue'); 
    ylim([-90,0]); xlim([-4.5,-0.5]);
    xlabel(sprintf("\\theta_{%d}", d-1)); 
    ylabel(sprintf("\\theta_{%d}", d)); 
    hold on;
    
    % randomly sample from accurate TT
    idx = randi([1, size(rosen_TT_samples, 1)], N);
    scatter(rosen_TT_samples(idx,d-1), rosen_TT_samples(idx,d), 5, 'red');
    ranks = num2str(ftt_rounded.r');
    % compute error to initial distribution in Frobenius
    fro_error = norm(ftt_rounded - ftt);
    title(join(["sum rank: ", sum(ranks), "error = ", fro_error]))
    fprintf(">>> Truncation accuracy = %s\n", ...
        fro_error);
    grid on;
end
end

% extremely truncated TT
ftt_rounded = round(ftt, 3); % 1 and lower works with normalizing flow correction
disp(join(['>>> truncation percentage = ', num2str(sum(ftt_rounded.r)/sum(ftt.r))]))

%[training_data, training_data_densities] = tt_irt_lin(gen_grid_arr, ...
%    ftt_rounded, unif_sample);

[training_data, training_data_densities] = tt_irt_debias(epoch*N, log_r_func, ftt_rounded, ...
        gen_grid_arr, 'mcmc', [true]);
disp(join([">>> training data generated, with shape: ", ...
    num2str(size(training_data))]))

% if ranks are all 1, plot the individual cores
if sum(ftt_rounded.r) == d+1
    disp("====== displaying individual cores ")
else
    disp("====== exiting ... ") 
end
if sum(ftt_rounded.r) == d+1
    % determine number of subplots
    num_subplots = ceil(sqrt(d+1));
    assert(num_subplots^2 >= d+1)
    figure(4); 
    for i = 1:d
        subplot(num_subplots,num_subplots,i); grid on; hold on;
        if i == d-1
            plot(x_dm1, ftt_rounded{i})
        elseif i == d
            plot(x_d, ftt_rounded{i})
        else
            plot(x_common, ftt_rounded{i})
        end
    end
end

% Rosenbrock sampler part IV: save data for pytorch NF
% output the sample points generated from 
% a high-d Rosenbrock (severely truncated)
% we use Python to construct NF

filename = '../data/tt_irt_rosen_sample.mat';
save(filename);