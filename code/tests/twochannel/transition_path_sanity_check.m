% sanity checks for Nonequilibrium Path Sampling
%% Check Energy Differences among Paths
clear; clc; rng('default');
% compare with 
% https://github.com/marylou-gabrie/flonaco/blob/main/flonaco

% check whether energy integral converges
N = 5:1000;
t_start = 0; t_end = 1; 
dt = (t_end - t_start) ./ (N + 1);
all_energy = zeros(1, length(N)); % upper canonical path
all_energy2 = all_energy;         % lower canonical path
all_energy3 = all_energy;         % middle path
beta = 1;
for factor = 0.8%0:0.01:(3/2)
    for i = 1:length(dt)
        dt_i = dt(i);
        N_i = N(i);
        % lower canonical path
        % spline (-1,0)->(0,-0.5)->(1,0)
        Nx = ceil(N_i/2);
        % first spline
        spline1 = linspace(-1,0,Nx);
        spline1_y = (-1/2) * spline1 - (1/2);
        % second spline
        spline2 = linspace(0,1,Nx);
        spline2_y = (1/2) * spline2 - (1/2);
        % whole path
        path_x = [spline1, spline2];
        path_y = [spline1_y, spline2_y];
        canonical_path = [path_x', path_y'];
        canonical_path = canonical_path(2:end-1,:);
        all_energy2(i) = (energy_potential(canonical_path(:),dt_i,beta));
        % spline (-1,0)->(-0.6,1.1)->(0,1.5)->(0.6,1.1)->(1,0)
        % canonical path: upper channel
        % first spline
        Nx = ceil(N_i/4);
        spline1 = linspace(-1,-0.6,Nx);
        spline1_y = (11/4)*spline1 + (11/4);
        % second spline
        spline2 = linspace(-0.6,0,Nx);
        spline2_y = (2/3)*spline2 + (3/2);
        % third spline
        spline3 = linspace(0,0.6,Nx);
        spline3_y = (-2/3)*spline3 + (3/2); 
        % fourth spline
        spline4 = linspace(0.6,1,Nx);
        spline4_y = (-11/4)*spline4 + (11/4); 

        % whole path
        canonical_path_x = [spline1, spline2, spline3, spline4];
        canonical_path_y = [spline1_y, spline2_y, spline3_y, spline4_y];
        canonical_path2 = [canonical_path_x', canonical_path_y'];
        % boundary xA, xB are already included in energy potential
        canonical_path2 = canonical_path2(2:end-1,:);
        % compute energy of upper channel
        all_energy(i) = (energy_potential(canonical_path2(:),dt_i,beta));

        % unlikely path, spline (-1,0)->(0,1/2)->(1,0)

        % first spline
        Nx = ceil(N_i/2);
        spline1 = linspace(-1,0,Nx);
        spline1_y = (factor) * spline1 + (factor);
        % second spline
        spline2 = linspace(0,1,Nx);
        spline2_y = (-factor) * spline2 + (factor);
        spline_x = [spline1, spline2];
        spline_y = [spline1_y, spline2_y];
        unlikely = [spline_x', spline_y'];
        unlikely = unlikely(2:end-1,:);
        all_energy3(i) = (energy_potential(unlikely(:),dt_i,beta));
    end
    factor
    figure(1); hold on;
    title(strcat("Path Action Comparison, factor = ", num2str(factor))); 
    grid on;
    plot(N, all_energy, 'LineWidth', 1.5, 'Color', 'blue');
    plot(N, all_energy2, 'LineWidth', 1.5, 'Color', 'red');
    plot(N, all_energy3, 'LineWidth', 1.5, 'Color', 'black');
    legend({'canonical path (upper)','canonical path (lower)', ...
        strcat('spline from [-1, 0] --> [0, ', num2str(factor), ']',...
        ' --> [1, 0]')})
end

%% Slices of PDF Contour Tensor
% use AMEN_CROSS to approximate the PDF tensor
% and plot slices (xi_1, xi_2) for i = 1, 2,..., N
clear; clc; rng('default');
% create function handle for PDF
N = 4;
X = sym('x', [N 2], 'real');
dt_all = linspace(1e-3, 2, 100);
k = 40; % spatial discretization

% vary beta
all_beta = 1e-6:1e-2:1; 

%disp(strcat("> Using temperature T = ", num2str(1/beta)));
L = 2;
grid = linspace(-L,L,k); % spatial domain
dx = grid(2)-grid(1);
grid_tt = tt_tensor(grid);
grid_tt_all = tt_meshgrid_vert(grid_tt, 2*N);
tmp = X(:).'; % symbolic variables
% create meshgrid for plotting
[Y_grid,X_grid] = meshgrid(grid);
% number of plots for square subplot
subplot_dim = ceil(sqrt(N));
% video file to export
make_movie = false; % true - makes a movie
if make_movie
    vidfile = VideoWriter('./img/pathPotentialPDF.mp4','MPEG-4');
    vidfile.FrameRate = 5;
    open(vidfile);
end

for idx = 1:length(all_beta) % idx = 1:length(dt_all)
    %dt = dt_all(idx);
    dt = 0.1;
    beta = all_beta(idx);
    f = figure(1);
        sgtitle(["Contour Plots of Marginal Distributions", ...
        sprintf("beta = %f, dt = %f ", beta, ...
        dt)]); 
    hold on;
    % create function handle
    pdf_func = exp(-0.25 * energy_potential(X(:), dt, beta));
    pdf_func = matlabFunction(pdf_func, 'Vars', {tmp});
    % amen_cross
    disp(strcat("===== Time Step dt = ", num2str(dt)))
    path_pdf_tt = amen_cross_s(grid_tt_all, pdf_func, 1e-10, 'nswp', 15);
    % contour plot of conditionals (X1_1, X1_2)
    % numerically integrate out all other dimensions
    weights = trapz_weights(k, dx).';
    %weights = ones(1,k).';
    % get cores
    all_cores = core(path_pdf_tt);
    % save all integrated cores
    % adjust shape
    all_cores_integrated = all_cores;
    all_cores_integrated{1} = reshape(all_cores_integrated{1}, ...
        [1 size(all_cores_integrated{1})]);
    all_cores_integrated{end} = permute(all_cores_integrated{end}, [2 1]);
    all_cores_integrated{end} = reshape(all_cores_integrated{end}, ...
        [size(all_cores_integrated{end}), 1]);
    for i = 2:length(all_cores_integrated)-1
        % permute, original core has shape n x r x r
        all_cores_integrated{i} = permute(all_cores_integrated{i}, ...
            [2 1 3]);
    end
    % numerical integration of the core along spatial dimension
    for i = 1:length(all_cores_integrated)
        all_cores_integrated{i} = tns_mult(all_cores_integrated{i}, ...
            2, weights, 1);
    end
    % contracting all integrated cores will yield normalization constant
    norm_c = all_cores_integrated{1};
    for i = 2:length(all_cores_integrated)
        norm_c = tns_mult(norm_c,2,all_cores_integrated{i},1);
    end
    % ========================
    % building marginal tensor
    % ========================
    % can reuse integrated cores to plot marginal of each X(i) 
    % for i = 1, 2, ..., N
    for i = 1:N
        % get unintegrated ones, should be only 2 dimensions
        marginal_cores = cell([2,1]);
        % contract with the rest of the cores
        if i == 1
            % special case
            marginal_cores{1} = reshape(all_cores{1}, ...
                [1 size(all_cores{1})]);
            marginal_core2 = permute(all_cores{2}, [2 1 3]);
            for j = 3:length(all_cores_integrated)
                % contract all integrated parts
                marginal_core2 = tns_mult(marginal_core2, 3, ...
                    all_cores_integrated{j},1);
            end
            marginal_cores{2} = marginal_core2;
        else
            marginal_core1 = all_cores_integrated{1};
            for j = 2:(2*i-2)
                marginal_core1 = tns_mult(marginal_core1, 2, ...
                    all_cores_integrated{j},1);
            end
            marginal_cores{1} = tns_mult(marginal_core1, 2, ...
                permute(all_cores{2*i-1}, [2 1 3]), 1);
            marginal_core2 = permute(all_cores{2*i}, [2 1 3]);
            for j = (2*i+1):length(all_cores_integrated)
                marginal_core2 = tns_mult(marginal_core2, 3, ...
                    all_cores_integrated{j},1);
            end
            marginal_cores{2} = marginal_core2;
        end
        % plotting routine + making movie
        % build marginal tensor of a single point
        marginal_tt = cell2core(tt_tensor, marginal_cores);
         % plot marginal tensor of point X_i
        marginal_tt_matrix = full(marginal_tt, [k k]);
        % normalize the matrix
        %marginal_tt_matrix = marginal_tt_matrix / sum(sum(marginal_tt_matrix));
        subplot(subplot_dim, subplot_dim, i)
        title(sprintf("Contour Plot of Point %d", i));
        xlabel(sprintf("X%d",i)); ylabel(sprintf("Y%d",i));
        hold on;
        contourf(X_grid, Y_grid, marginal_tt_matrix); colorbar; 
        shading interp;
    end
        if make_movie
        % save movie
        frame = getframe(gcf);
        writeVideo(vidfile, frame);
        clf(f);
    end
end
if make_movie
    close(vidfile);
end

%% Conditional Sampling from Discrete Tensor
% implement conditional distribution sampling from
% the discrete grid points (no interpolation as described
% in TT-IRT)
clear; clc; rng('default');
% parameters
N = 20;
X = sym('x', [N 2], 'real');
dt = 1.5e-4;
K = 200; % spatial discretization
beta = 0.01; 
disp(strcat("> Using temperature T = ", num2str(1/beta)));
L = 2;
grid = linspace(-L,L,K); % spatial domain
dx = grid(2)-grid(1);
grid_tt = tt_tensor(grid);
grid_tt_all = tt_meshgrid_vert(grid_tt, 2*N);
tmp = X(:).'; % symbolic variables

% create TT from AMEN_CROSS
% create function handle
pdf_func = exp(-0.25 * energy_potential(X(:), dt, beta));
pdf_func = matlabFunction(pdf_func, 'Vars', {tmp});
% amen_cross
disp(strcat("===== Time Step dt = ", num2str(dt)))
path_pdf_tt = amen_cross_s(grid_tt_all, pdf_func, 1e-10, 'nswp', 15);
% numerically integrate out all other dimensions
weights = trapz_weights(K, dx).';
all_cores = core(path_pdf_tt);
% prepare integrated cores
all_cores_integrated = all_cores;
all_cores_integrated{1} = reshape(all_cores_integrated{1}, ...
    [1 size(all_cores_integrated{1})]);
all_cores_integrated{end} = permute(all_cores_integrated{end}, [2 1]);
all_cores_integrated{end} = reshape(all_cores_integrated{end}, ...
    [size(all_cores_integrated{end}), 1]);
for i = 2:length(all_cores_integrated)-1
    % permute, original core has shape n x r x r
    all_cores_integrated{i} = permute(all_cores_integrated{i}, ...
        [2 1 3]);
end
% numerical integration of the core along spatial dimension
for i = 1:length(all_cores_integrated)
    all_cores_integrated{i} = tns_mult(all_cores_integrated{i}, ...
        2, weights, 1);
end
% contracting all integrated cores will yield normalization constant
norm_c = all_cores_integrated{1};
for i = 2:length(all_cores_integrated)
    norm_c = tns_mult(norm_c,2,all_cores_integrated{i},1);
end

% number of independent samples
M = 2^9;
% conditional sampling from the grid
% save sampled coordinates, each row is a sample (x1,x2,...,xN)
save_sample = zeros(M,2*N); 
save_index = zeros(M,2*N);
for i = 1:2*N
    if i == 1
        % sampling first coordinate
        % only need the marginal p(x1_1)
        marginal_cores = all_cores_integrated;
        % only replace the first core to be variable, then
        % contract all rest
        marginal_cores{1} = ...
            reshape(all_cores{1}, [1,size(all_cores{1})]);
        marginal_tt = marginal_cores{1};
        for j = 2:2*N
            marginal_tt = tns_mult(marginal_tt,3,marginal_cores{j},1);
        end
        % take absolute value of marginal
        %marginal_tt = abs(marginal_tt);
        
        % set negative values to 0
        neg_vals = find(marginal_tt<0);
        marginal_tt(neg_vals) = 0;
        
        % sampling is randsample from the spatial grid 
        % with weight as the marginal (unnormalized)
        
        % sample an index
        coord_index = randsample(length(grid),M,true,marginal_tt);
        save_index(:,i) = coord_index;
        % sampled points
        coord_samples = grid(coord_index);
        save_sample(:,i) = coord_samples;
    else
        % general case
        % need the k-th marginal p(x1_1,x1_2,...,xk)
        % replace all 1,2,...k-th cores as variable
        marginal_cores = all_cores_integrated;
        for j = 1:i
            if j == 1
                % first core, has shape (n x r)
                marginal_cores{j} = ...
                    reshape(all_cores{j},[1,size(all_cores{j})]);
            elseif j == 2*N
                % last core, has shape (n x r)
                marginal_cores{j} = ...
                    permute(all_cores{j}, [2 1]);
            else
                % middle cores, has shape (n x r x r)
                marginal_cores{j} = permute(all_cores{j}, [2 1 3]);  
            end
        end
        % contract cores (i+1,i+2,...,2*N)
        tmp = marginal_cores{i};
        for j = (i+1):2*N
            % contract
            tmp = tns_mult(tmp,3,marginal_cores{j},1);
        end
        % put back as core i
        marginal_cores{i} = tmp;
        marginal_cores = marginal_cores(1:i); % only need the first i cores
        
        % sampling step
        for j = 1:M
            tmp_marginal = marginal_cores;
            % index out (1,2,...,i-1)
            for k = 1:(i-1)
                % get indices for coordinate k
                tmp = save_index(:,k);
                % get index for j-th sample of coordinate k
                idx = tmp(j);
                % index out coordinate k
                tmp = tmp_marginal{k};
                tmp_marginal{k} = tmp(:,idx,:);
            end
            % contract to obtain k-th marginal p(x1_1,x1_2,...,xk)
            marginal_tt = tmp_marginal{1};
            for l = 2:i
                % adjust shape
                marginal_tt = squeeze(marginal_tt).';
                marginal_tt = reshape(marginal_tt, ...
                    [1 1 length(marginal_tt)]);
                marginal_tt = tns_mult(marginal_tt,3,tmp_marginal{l},1);
            end
            % take absolute value of marginal
            %marginal_tt = abs(squeeze(marginal_tt));
            
            % set negative values to 0
            marginal_tt = squeeze(marginal_tt);
            neg_vals = find(marginal_tt<0);
            marginal_tt(neg_vals) = 0;
            
            % sample from the k-th marginal
            save_index(j,i) = randsample(length(grid),1,true,marginal_tt);
            save_sample(j,i) = grid(save_index(j,i));
        end
    end
    figure(2);
    hold on
    % normalize and plot marginal distribution
    marginal_tt = marginal_tt / sum(marginal_tt);
    plot(grid, marginal_tt)
end

% plot sampled points from discrete tensor
for i = 1:M
    % get a path sample
    path = reshape(save_sample(i,:), [N 2]);
    % plot
    figure(1); 
    hold on
    plot(path(:,1),path(:,2));
end

%% Gibbs Sampling of R Points
clear; clc; rng('default');
% Perform Gibbs Sampling on the discrete tensor train.
% Do it in two ways:
% 1. use exact evaluation
% 2. use tensor train contracted cores

% ============================
%  Method 1: exact evaluation
% ============================
% parameters
N = 10;
X = sym('x', [N 2], 'real');
dt = 6e-3;
K = 500; % spatial discretization
beta = 1; 
disp(strcat("> Using temperature T = ", num2str(1/beta)));
L = 2;
grid = linspace(-L,L,K); % spatial domain
dx = grid(2)-grid(1);
tmp = X(:).'; % symbolic variables
disp(strcat("===== Time Step dt = ", num2str(dt)))

% Gibb's sampler with exact evalution
supp = grid;
M = 2^10; % number of samples

% each row represents a sample
samples = zeros(M,2*N); 
samples_ind = zeros(M,2*N); 

% initial sample, take a path from (-1,0) to (1,0)
% the variable ordering is in accordance with TMP
X0 = zeros(N,2); 
X0(:,1) = linspace(-1,1,N);

X0 = zeros(N, 2);
X0(1:round(N/2), :) = [linspace(-1, 0, round(N/2)).', ...
    ((3/2)*linspace(-1, 0, round(N/2)) + 3/2).'];
X0(round(N/2)+1:N, :) = [linspace(0, 1, N-round(N/2)).', ...
    ((-3/2)*linspace(0, 1, N-round(N/2)) + 3/2).'];

% set the first row which will be updated in the first iteration
samples(1,:) = X0(:); 
for i = 1:M
    i
    for j = 1:2*N  
        % for each j, we need to build a probability vector
        % which is evaluated from joint where x1,x2,...,xj-1
        % and xj+1,...,xd are fixed, but xj is variable.
        prob = zeros(1,K);
        for idx = 1:K
            % loop over all possible values of xj and fill prob vector
            path_with_x_j = samples(i,:);
            % replace the j-th position with possible value
            % while holding all other data fixed
            path_with_x_j(j) = grid(idx);
            prob(idx) = exp(-0.25*...
                energy_potential(path_with_x_j, dt, beta));
        end
        % normalize to a prob vector
        prob = prob / sum(prob); 
        figure(1); hold on
        plot(prob)
        % draw index using the prob vector
        samples_ind(i,j) = randsample(K,1,true,prob);
        % update the j-th coordinate with the exact value from the support
        samples(i,j) = grid(samples_ind(i, j)); 
    end
    if i < N
        % set the next row with the current sample 
        % which will be updated in the following iteration
        samples(i+1,:) = samples(i,:); 
    end
end

% plot samples
for i = 1:M
    path = reshape(samples(i,:), [N 2]);
    figure(1);
    hold on;
    plot(path(:,1), path(:,2));
end
% ============================
%  Method 2: tensor evaluation
% ============================


%% Gibbs Sampling of R^2 Points
clear; clc; rng('default');
% Perform Gibbs Sampling on the discrete tensor train
% where each core represents a grid in R^4.


% ==================================
%  Task 1: Build SVD TT of length N
% ==================================
% parameters
L = 2;
beta = 0.01; 
temperature = 1/beta;
dt = 1e-3;
c = 2.5;
N = 50; % effective number of time discretization
N_ = N+2; % effective N used for computations (xA, xB included as variable)
t_end = dt*N;


k = 29; % k = 5,13,17,21,25
grid = linspace(-L,L,k);
xA = [-1,0]; xB = [1,0]; % fixed boundary points
assert(ismember(xA(1), grid) & ismember(xA(2), grid), ...
        "boundary points need to be included in the grid"); 
    assert(ismember(xB(1), grid) & ismember(xB(2), grid), ...
        "boundary points need to be included in the grid"); 
dx = grid(2)-grid(1); x_start = grid(1); x_end = grid(end);
% column major indexer into 2d grid [-L,L]x[-L,L]
idx = 1:(k^2);
[X,Y] = meshgrid(grid); % column major points
all_points = [X(:) Y(:)];
[X,Y] = meshgrid(1:k); % column major indexer
all_idx = [X(:) Y(:)];
% find flattened indices for xA and xB
xA_idx1 = find(grid==xA(1));
xA_idx2 = find(grid==xA(2));
xB_idx1 = find(grid==xB(1));
xB_idx2 = find(grid==xB(2));
xA_double_idx = [xA_idx1, xA_idx2];
xB_double_idx = [xB_idx1, xB_idx2];
xA_composite_idx = find(ismember(all_idx, xA_double_idx, 'rows'));
xB_composite_idx = find(ismember(all_idx, xB_double_idx, 'rows'));
% verify that composite index does correspond to xA and xB
assert(all(all_points(xA_composite_idx, :)==xA), "xA is not located")
assert(all(all_points(xB_composite_idx, :)==xB), "xB is not located")

clear X Y;
% compute drift
all_drift = b(all_points, c);
% form kernels, there are two different though similar kernels
% - Vars: xA, x1, x2, ..., xN, xB
% - xA Boundary Kernel: (1/dt)*(X(1,:) - xA) + b(X(1,:))
% - Common Kernel:      (1/dt)*(X(i+1,:) - X(i,:)) + b(X(i,:))

% xA kernel
xA_energy_row = ...
        ( all_points(:,1) - dt * all_drift(:,1) ).' - ...
        all_points(:,1); % size k^2 x k^2
xA_energy_col = ...
    ( all_points(:,2) - dt * all_drift(:,2) ).' - ...
    all_points(:,2);
K_xA = exp( -0.25 .* ( beta/dt ) .* ...
    ( xA_energy_row.^2 + xA_energy_col.^2 ) );

% common kernel
energy_row = ...
        all_points(:,1).'...
            - (all_points(:,1) + dt * all_drift(:,1)); % size k^2 x k^2
energy_col = ...
    all_points(:,2).'...
        -(all_points(:,2) + dt * all_drift(:,2));
K = energy_row.^2 + energy_col.^2;                 
K = ( beta/dt ) * K;
K = exp(-0.25 * K);

% perform SVD
[u1,s1,v1] = svd(K_xA);
a1 = u1*s1; b1 = v1;
[u2,s2,v2] = svd(K);
a2 = u2*s2; b2 = v2;
% truncate
svd_tol = 1e-3; % tolerance for SVD truncation
idx1 = find(diag(s1)./max(diag(s1))>svd_tol);
idx2 = find(diag(s2)./max(diag(s2))>svd_tol);
r1 = length(idx1); r2 = length(idx2);
a1_r = a1(:,idx1); b1_r = b1(:,idx1);
a2_r = a2(:,idx2); b2_r = b2(:,idx2);
disp(strcat("> Kernel SVD Norm Error, K_xA = ", ...
    num2str(norm(a1_r*b1_r'-K_xA)), ...
    ", K = ", num2str(norm(a2_r*b2_r'-K))));
% build cores
xA_core = zeros(r1,k^2,r2);
common_core = zeros(r2,k^2,r2);
for i = 1:r1
    for j = 1:r2
        xA_core(i,:,j) = b1_r(:,i).*a2_r(:,j);
    end
end
for i = 1:r2
    for j = 1:r2
        common_core(i,:,j) = b2_r(:,i).*a2_r(:,j);
    end
end
left_core = reshape(a1_r, [1 size(a1_r)]);
right_core = reshape(b2_r', [size(b2_r') 1]);
assert(N_>3, "supports sampling >=2 points in R^2") % xA, x1, x2, ..., xB
core_cells = cell(N_,1);
core_cells{1} = left_core;
core_cells{2} = xA_core;
core_cells{end} = right_core;
for i = 3:(N_-1)
    % repeat common core
    core_cells{i} = common_core;
end
% Tensor Train usesd for Gibbs Sampling
% should be ordered as xA, x1, x2, ..., xB
% ==============
%  Test ordering
% ==============
if (N_ == 4) && (k <= 13)
    % build exact TT and test accuracy
    exact_tensor = zeros(k^2,k^2,k^2,k^2);
    % finish ...
end

svd_tt = cell2core(tt_tensor, core_cells);
% index out xA and xB (xA_composite_idx, xB_composite_idx)
indexer_ = cell(1,N_);
indexer_{1} = xA_composite_idx;
indexer_{end} = xB_composite_idx;
for i = 2:(N_-1)
    indexer_{i} = ':';
end
svd_tt = svd_tt(indexer_); clear indexer_;

% ==============================================
%  Task 2: Perform Gibbs Sampling on the SVD TT
% ==============================================
% numerically integrate out all other dimensions
% the weights should be trapz weights of 2D then
% flattened columnwise
weights = trapz_weights(k,dx)' .* trapz_weights(k,dx); 
weights = weights(:);
all_cores = core(svd_tt);
% prepare integrated cores
all_cores_integrated = all_cores;
all_cores_integrated{1} = reshape(all_cores_integrated{1}, ...
    [1 size(all_cores_integrated{1})]);
all_cores_integrated{end} = permute(all_cores_integrated{end}, [2 1]);
all_cores_integrated{end} = reshape(all_cores_integrated{end}, ...
    [size(all_cores_integrated{end}), 1]);
for i = 2:length(all_cores_integrated)-1
    % permute, original core has shape n x r x r
    all_cores_integrated{i} = permute(all_cores_integrated{i}, ...
        [2 1 3]);
end
% numerical integration of the core along spatial dimension
for i = 1:length(all_cores_integrated)
    all_cores_integrated{i} = tns_mult(all_cores_integrated{i}, ...
        2, weights, 1);
end
% contracting all integrated cores will yield normalization constant
norm_c = all_cores_integrated{1};
for i = 2:length(all_cores_integrated)
    norm_c = tns_mult(norm_c,2,all_cores_integrated{i},1);
end
% number of independent samples
M = 2^9;
% conditional sampling from the grid
% save sampled coordinates, each row is a sample (x1,x2,...,xN)
save_index = zeros(M,N);
for i = 1:N
    if i == 1
        % sampling first coordinate
        % only need the marginal p(x1_1)
        marginal_cores = all_cores_integrated;
        % only replace the first core to be variable, then
        % contract all rest
        marginal_cores{1} = ...
            reshape(all_cores{1}, [1,size(all_cores{1})]);
        marginal_tt = marginal_cores{1};
        for j = 2:N
            marginal_tt = tns_mult(marginal_tt,3,marginal_cores{j},1);
        end
        % take absolute value of marginal
        %marginal_tt = abs(marginal_tt);
        
        % set negative values to 0
        neg_vals = find(marginal_tt<0);
        marginal_tt(neg_vals) = 0;
        
        % sampling is randsample from the spatial grid 
        % with weight as the marginal (unnormalized)
        
        % sample an index
        coord_index = randsample(length(grid)^2,M,true,marginal_tt);
        save_index(:,i) = coord_index;
    else
        % general case
        % need the k-th marginal p(x1_1,x1_2,...,xk)
        % replace all 1,2,...k-th cores as variable
        marginal_cores = all_cores_integrated;
        for j = 1:i
            if j == 1
                % first core, has shape (n x r)
                marginal_cores{j} = ...
                    reshape(all_cores{j},[1,size(all_cores{j})]);
            elseif j == N
                % last core, has shape (n x r)
                marginal_cores{j} = ...
                    permute(all_cores{j}, [2 1]);
            else
                % middle cores, has shape (n x r x r)
                marginal_cores{j} = permute(all_cores{j}, [2 1 3]);  
            end
        end
        % contract cores (i+1,i+2,...,2*N)
        tmp = marginal_cores{i};
        for j = (i+1):N
            % contract
            tmp = tns_mult(tmp,3,marginal_cores{j},1);
        end
        % put back as core i
        marginal_cores{i} = tmp;
        marginal_cores = marginal_cores(1:i); % only need the first i cores
        
        % sampling step
        for j = 1:M
            tmp_marginal = marginal_cores;
            % index out (1,2,...,i-1)
            for k = 1:(i-1)
                % get indices for coordinate k
                tmp = save_index(:,k);
                % get index for j-th sample of coordinate k
                idx = tmp(j);
                % index out coordinate k
                tmp = tmp_marginal{k};
                tmp_marginal{k} = tmp(:,idx,:);
            end
            % contract to obtain k-th marginal p(x1_1,x1_2,...,xk)
            marginal_tt = tmp_marginal{1};
            for l = 2:i
                % adjust shape
                marginal_tt = squeeze(marginal_tt).';
                marginal_tt = reshape(marginal_tt, ...
                    [1 1 length(marginal_tt)]);
                marginal_tt = tns_mult(marginal_tt,3,tmp_marginal{l},1);
            end
            % take absolute value of marginal
            %marginal_tt = abs(squeeze(marginal_tt));
            
            % set negative values to 0
            marginal_tt = squeeze(marginal_tt);
            neg_vals = find(marginal_tt<0);
            marginal_tt(neg_vals) = 0;
            
            % sample from the k-th marginal
            save_index(j,i) = randsample(length(grid)^2,1,true,marginal_tt);
        end
    end
end

% get coordinates based on indices (1 <= i <= k^2)
sample_points = zeros(M,N,2);
for i = 1:M
    path_indices = save_index(i,:);
    for j = 1:N
        % get index for each point
        point_j_idx = path_indices(j);
        % convert index into multi-index for 2d grid
        point_j_multiidx = all_idx(point_j_idx,:);
        % convert index into coordinates in 2d grid and save
        point_j = [grid(point_j_multiidx(1)), grid(point_j_multiidx(2))];
        sample_points(i,j,:) = point_j;
    end
end

%% Test approx_transition_kernel.m works correctly
clear; clc; rng('default');
% check approx_transition_kernel and amen_cross gives similar tensors
t_start = 0; 
beta = 0.01;
N = 20;
dt = 1.5e-2;
t_end = t_start + N*dt;


% spatial domain [-L,L]^d
L = 2; 
k = 13;
X_arr = linspace(-L, L, k);
dx = X_arr(2)-X_arr(1);

% make sure boundary points are contained in X_arr
xA = [-1,0]; xB = [1,0];
assert(ismember(xA(1), X_arr) & ismember(xA(2), X_arr), ...
        "boundary points need to be included in the grid"); 
    assert(ismember(xB(1), X_arr) & ismember(xB(2), X_arr), ...
        "boundary points need to be included in the grid");
% ==============
%  Sanity Check
% ==============

% similar to Test 3 from previous section
% test norm error btwn SVD TT and AMEN_CROSS TT

% SVD approach to get a TT
path_pdf_func_svd_tt = approx_transition_kernel(X_arr, N, 1e-20, dt, beta);

% AMEN_CROSS to get a TT (align variable ordering with SVD)
% (xN-1_2, xN-1_1, ..., x2_2, x2_1, x1_2, x1_1)
X = sym('x', [N 2], 'real');
% create exact function for AMEN_CROSS
func_sym = exp(-0.25 * energy_potential(X(:), dt, beta));
tmp = X(:).';
tmp = reshape(tmp, [N 2]).';
tmp = flip(tmp(:)).';
func = matlabFunction(func_sym, 'Vars', {tmp});
% create AMEN_CROSS tensor
X_tt = tt_tensor(X_arr); 
X = tt_meshgrid_vert(X_tt, 2*N);
test_tt = amen_cross_s(X, func, 1e-10, 'nswp', 30);
disp(strcat("===== error btwn (SVD TT, AMEN_CROSS TT (reordered) ) = ", ...
    num2str(norm(path_pdf_func_svd_tt - test_tt))))

% similar to Test 1&2 from previous section
% test norm error btwn SVD TT and exact tensor (d<4 in order to be cheap)

if N == 3
    exact_tensor = zeros(k,k,k,k,k,k);
    for i1 = 1:k
        for i2 = 1:k
            for i3 = 1:k
                for i4 = 1:k
                    for i5 = 1:k
                        for i6 = 1:k
                            x1_1 = X_arr(i1);
                            x1_2 = X_arr(i2);
                            x2_1 = X_arr(i3);
                            x2_2 = X_arr(i4);
                            x3_1 = X_arr(i5);
                            x3_2 = X_arr(i6);
                            path = [x1_1, x1_2; ...
                                    x2_1, x2_2; ...
                                    x3_1, x3_2; ...
                                    ];
                            exact_tensor(i6,i5,i4,i3,i2,i1) = ...
                              exp(-0.25 ...
                              * energy_potential(path(:), dt, beta)...
                              );
                        end
                    end
                end
            end
        end
    end
    % check norm error between AMEN_CROSS and exact
    disp(strcat("===== error btwn (AMEN TT, exact tensor) = ", ...
       num2str(norm(full(test_tt)-exact_tensor(:))) ))
    % check norm error between SVD TT and exact
    disp(strcat("===== error btwn (SVD TT, exact tensor) = ", ...
       num2str(norm(full(path_pdf_func_svd_tt)-exact_tensor(:))) ))
end



for i = 1:M
    figure(2); 
    title(sprintf("Nonequilibrium Path Samples from SVD, N = %d", ...
        i));
    % ordering from SVD_TT:
    % X = (xN-1_2, xN-1_1, ..., x2_2, x2_1, x1_2, x1_1)
    path_i = flip(path_samples_svd(i,:));
    path_i = reshape(path_i, [2 N]).';
    hold on;
    % add boundary points
    path_i = [xA; path_i; xB];
    plot(path_i(:,1), path_i(:,2), 'LineWidth', 1.2);
end

%% (NP) Energy contour of a single point 
% NP: a single point / single discretization is not informative. 
% suppose we only sampled 1 point, with boundary conditions, 
% look at the energy to determine the likely regions
clear; clc; rng('default');
k = 100; L = 2;
x_start = -L; x_end = L;
grid = linspace(x_start, x_end, k);
[X,Y] = meshgrid(grid);
% column-wise flatten
all_points = [X(:) Y(:)];
beta = 1;
temperature = 1/beta;
if temperature >= 100
    disp(strcat("> Using High Temperature, T = ", ...
        num2str(temperature), " , Lower channel is preferred. "))
elseif temperature <= 1
    disp(strcat("> Using Low Temperature, T = ", ...
        num2str(temperature), " , Upper channel is preferred. "))
else
    disp(strcat("> Using Temperature, T = ", ...
        num2str(temperature)))
end

% plot energy contour with changing dt
Z = zeros(1, k^2);
Z_energy = zeros(1, k^2);
all_dt = (linspace(1e-4, 0.3, 100));
for dt = all_dt
    for i = 1:(k^2)
        single_point = all_points(i, :);
        % compute PDF of [xA, x, xB]
        %Z(i) = exp(-0.25*energy_potential(single_point, dt, beta));
        Z(i) = energy_potential(single_point, dt, beta);
    end
    Z_grid = reshape(Z, [k k]);
    Z_energy_grid = reshape(Z_energy, [k k]);
    Z_energy_grid = Z_energy_grid ./ sum(sum(Z_energy_grid));
    % plot energy contour
    figure(1);
    surfc(X,Y,Z_grid); 
    %contourf(X,Y,Z_grid); 
    colorbar; 
    %sgtitle(strcat("[ PDF Contour ] Range of Energy: ", ...
    %    "(min) = ", num2str(min(Z)), " (max) = ", num2str(max(Z))));
    subtitle(strcat("| dt = ", num2str(dt), " | Upper Channel Energy = ", ...
        num2str(energy_potential([0,3/2], dt, beta)), ...
        ", Lower Channel Energy = ", ...
        num2str(energy_potential([0,0], dt, beta)))); 
end

