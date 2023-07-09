% Unit tests with SVD Tensor Train using Transition Path Energy
clc; clear;
rng('default');

% Test 1: (2d, N = 3) exact tensor vs SVD
%       exp( -( |(X_3 - X_2)/dt - sin(X_2)|^2  + 
%               |(X_2 - X_1)/dt - sin(X_1)|^2 
k = 21;

% spatial grid in 2d, for each direction
dt = 1;
x_grid = linspace(-1,1,k);
[X_grid, Y_grid] = meshgrid(x_grid);
% flattened column-major
%[~, ~, ~, row_coord, col_coord] = flatten2d(x_grid);
all_points = [X_grid(:), Y_grid(:)];
% create symbolic var for exact computation
X = sym('x', [3,2]);
func_sym = exp(-( sum( ( ( X(3,:)-X(2,:) )/dt - sin( X(2,:) ) ).^2 ) + ...
           sum( ( ( X(2,:)-X(1,:) )/dt - sin( X(1,:) ) ).^2 ) ));
% convert to callable function
func_sym = matlabFunction(func_sym);

% build exact tensor
run_exact = true; % save time for quick runs
if run_exact
    exact_tt = zeros(k,k,k,k,k,k);
    for i1 = 1:k
        x1_1 = x_grid(i1);
        for i2 = 1:k 
            x1_2 = x_grid(i2);
            for i3 = 1:k
                x2_1 = x_grid(i3);
                for i4 = 1:k
                    x2_2 = x_grid(i4);
                    for i5 = 1:k
                        x3_1 = x_grid(i5);
                        for i6 = 1:k
                            x3_2 = x_grid(i6);
                            exact_tt(i6,i5,i4,i3,i2,i1) = ...
                                func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                            %exact_tt(i1,i2,i3,i4,i5,i6) = ...
                            %    func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                        end
                    end
                end
            end
        end
    end
end
% build SVD tensor train
x_all = all_points(:,1); 
y_all = all_points(:,2);
% (1/dt^2) * |X_i - ( X_i-1 + dt * sin(X_i-1) )|^2
% build kernel
K_x = x_all' - (x_all + dt * sin(x_all));
K_y = y_all' - (y_all + dt * sin(y_all));

%K_x = x_all - (x_all + dt * sin(x_all))';
%K_y = y_all - (y_all + dt * sin(y_all))';
K = K_x.^2 + K_y.^2;
K = exp(-(1/dt^2) * K);

% first SVD to separate X1, X2, X3
tol = 1e-16;
[A,B,r] = interp_decomp(K,tol);
    
% (k^2 x r) * (r x k^2) 
common_core = zeros(r,k^2,r);
for i = 1:r
    for j = 1:r
        common_core(i,:,j) = A(:,i).*B(:,j);
    end
end

% second SVD to decouple x, y coords
common_core_reshape = reshape(common_core, [r*k, k*r]);
% (r*k x r2) * (r2 x k*r)
[A2,B2,r2] = interp_decomp(common_core_reshape,tol);
common_core_x = reshape(A2,[r k r2]);
common_core_y = reshape(B2',[r2,k,r]);
% boundary cores
left_boundary = reshape(B, [1 k^2 r]); % (1 x k^2 x r)
left_boundary = reshape(left_boundary, [1*k k*r]);
% (k x r_left) * (r_left x k*r)
[A_left,B_left,r_left] = interp_decomp(left_boundary,tol);
core_left1 = reshape(A_left, [1 k r_left]);
core_left2 = reshape(B_left', [r_left k r]);
    
right_boundary = reshape(A', [r k^2 1]); % (r x k^2 x 1)
right_boundary = reshape(right_boundary, [r*k k*1]);
% (r*k x r_right) * (r_right x k*1)
[A_right,B_right,r_right] = interp_decomp(right_boundary,tol);
core_right1 = reshape(A_right, [r k r_right]);
core_right2 = reshape(B_right', [r_right k 1]);
% form tensor train
core_cells = {core_left1; core_left2; ...
              common_core_x; common_core_y; ...
              core_right1; core_right2};

svd_tt = cell2core(tt_tensor, core_cells);
disp("> Norm error between ( Exact ) and ( SVD ), should expect macheps. \n")
norm(full(svd_tt)-exact_tt(:))

%%
% Test 2: investigate whether truncated SVD of matrix transpose is different
% from SVD of a matrix itself
clc; clear;
rng("default");

tol = 1e-16;
% generate random, large matrix for testing
k = 50;
K = sin(randn(k,k));
K_transpose = K';
% SVD of K
[A,B,r] = interp_decomp(K, tol);

% SVD of K^T
[A,B,r] = interp_decomp(K_transpose, tol);
disp(strcat("> ========== Error from interpolative SVD = ", ...
        num2str(norm((B*A') - K))))
    
% run many trials of SVD on random matrix, scatter plot
% the condition number with SVD reconstruction precision
mc_trials = 1e+4;
svd_precision = zeros([1,mc_trials]);
matrix_conds = zeros([1,mc_trials]);
for i = 1:mc_trials
    i
    % generate random matrix
    K = rand(k,k);
    % compute svd
    [u,s,v] = svd(K);
    % record reconstruction error
    svd_precision(i) = norm(u*s*v'-K);
    % record random matrix condition number
    matrix_conds(i) = norm(K)*norm(inv(K));
end

% scatter plot
figure(1);
sz = 25;
c = linspace(1,10,length(svd_precision));
scatter(log10(matrix_conds),log10(svd_precision),sz,c,'filled');

% run many trials of SVD on random matrix, compare the reconstruction
% errors of SVD(K) and SVD(K^T)
mc_trials = 1e+4;
svd_precision = zeros([1,mc_trials]);
svd_transpose_precision = zeros([1,mc_trials]);
for i = 1:mc_trials
    i
    % generate random matrix
    K = rand(k,k);
    % compute svd
    [u,s,v] = svd(K);
    % record reconstruction error
    svd_precision(i) = norm(u*s*v'-K);
    % record recons error of transpose(K)
    [u,s,v] = svd(K');
    svd_transpose_precision(i) = norm(u*s*v'-K');
end

% scatter plot
figure(2);
sz = 25;
c = linspace(1,10,length(svd_precision));
scatter(log10(svd_precision),log10(svd_transpose_precision),sz,c,'filled');

%% Test 3, R^2 points with indexing and different boundary
clc; clear;
rng('default');

% Test 3: (2d, N = 3) exact tensor vs SVD
%       exp( -( |(xB  - X_3)/dt - b(X_3)|^2  +
%               |(X_3 - X_2)/dt - b(X_2)|^2  + 
%               |(X_2 - X_1)/dt - b(X_1)|^2  +
%               |(X_1 - xA )/dt - b(X_1)|^2
k = 13;
c = 2.5; % drift
% spatial grid in 2d, for each direction
dt = 100;
x_grid = linspace(-2,2,k);

% boundary points
xA = [-1, 0];
xB = [1, 0];
% check if points are able to be indexed later in grid
assert(isin_grid(xA, x_grid)); 
assert(isin_grid(xB, x_grid));
% flatten grid, same as meshgrid
[all_idx, row_indexer, col_indexer, row_coord, col_coord] = ...
    flatten2d(x_grid);
all_points = [row_coord(:), col_coord(:)];
% for indexing
[xA_xpos, xA_ypos] = find_idx2d(xA, x_grid);
[xB_xpos, xB_ypos] = find_idx2d(xB, x_grid);

% create symbolic var for exact computation
X = sym('x', [3,2]);
func_sym = exp(-( ...
           sum( ( ( xB-X(3,:) )/dt -  noneq_drift(X(3,:), c) ).^2 ) + ...
           sum( ( ( X(3,:)-X(2,:) )/dt - noneq_drift(X(2,:), c) ).^2 ) + ...
           sum( ( ( X(2,:)-X(1,:) )/dt - noneq_drift(X(1,:), c) ).^2 ) + ...
           sum( ( ( X(1,:)-xA )/dt - noneq_drift(X(1,:), c) ).^2 ) ...
       ));
% convert to callable function
func_sym = matlabFunction(func_sym);

% build exact tensor
run_exact = true; % save time for quick runs
if run_exact
    exact_tt = zeros(k,k,k,k,k,k);
    for i1 = 1:k
        i1
        x1_1 = x_grid(i1);
        for i2 = 1:k 
            x1_2 = x_grid(i2);
            for i3 = 1:k
                x2_1 = x_grid(i3);
                for i4 = 1:k
                    x2_2 = x_grid(i4);
                    for i5 = 1:k
                        x3_1 = x_grid(i5);
                        for i6 = 1:k
                            x3_2 = x_grid(i6);
                            exact_tt(i6,i5,i4,i3,i2,i1) = ...
                                func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                            %exact_tt(i1,i2,i3,i4,i5,i6) = ...
                            %    func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                        end
                    end
                end
            end
        end
    end
end

% build SVD tensor train
% create two additional cores for xA, xB, then index out
x_all = all_points(:,1); 
y_all = all_points(:,2);
% (1/dt^2) * |X_i - ( X_i-1 + dt * sin(X_i-1) )|^2
% build kernel K(X_1, X_2), K(X_2, X_3), K(X_3, xB)
all_drift = noneq_drift(all_points, c);
K_x = x_all' - (x_all + dt * all_drift(:,1) );
K_y = y_all' - (y_all + dt * all_drift(:,2) );

%K_x = x_all - (x_all + dt * sin(x_all))';
%K_y = y_all - (y_all + dt * sin(y_all))';
K = K_x.^2 + K_y.^2;
K = exp(-(1/dt^2) * K);

% first SVD to separate X1, X2, X3, xB
tol = 1e-16;
[A,B,r] = interp_decomp(K,tol);
    
% (k^2 x r) * (r x k^2) 
common_core = zeros(r,k^2,r);
for i = 1:r
    for j = 1:r
        common_core(i,:,j) = A(:,i).*B(:,j);
    end
end

% second SVD to decouple x, y coords
common_core_reshape = reshape(common_core, [r*k, k*r]);
% (r*k x r2) * (r2 x k*r)
[A2,B2,r2] = interp_decomp(common_core_reshape,tol);
common_core_x = reshape(A2,[r k r2]);
common_core_y = reshape(B2',[r2,k,r]);

% left boundary cores
left_boundary = reshape(B, [1 k^2 r]); % (1 x k^2 x r)
left_boundary = reshape(left_boundary, [1*k k*r]);
% (k x r_left) * (r_left x k*r)
[A_left,B_left,r_left] = interp_decomp(left_boundary,tol);
core_left1 = reshape(A_left, [1 k r_left]);
core_left2 = reshape(B_left', [r_left k r]);

% right boundary and first core are different
% (1/dt^2) * | ( X_1 - dt * sin(X_1) ) - ( xA ) |^2
% build kernel K(xA, X_1)
K_x_xA = ( x_all - dt * all_drift(:,1) )' - ( x_all );
K_y_xA = ( y_all - dt * all_drift(:,2) )' - ( y_all );
K_xA = K_x_xA.^2 + K_y_xA.^2;
K_xA = exp(-(1/dt^2) * K_xA);

[A_tilde,B_tilde,r_tilde] = interp_decomp(K_xA,tol);
% (k^2 x r_tilde) * (r_tilde x k^2) 
x1_core = zeros(r,k^2,r_tilde);
for i = 1:r
    for j = 1:r_tilde
        x1_core(i,:,j) = A(:,i).*B_tilde(:,j);
    end
end

% second SVD to decouple X_1(x), X_1(y)
x1_core_reshape = reshape(x1_core, [r*k, k*r_tilde]);
% (r*k x r2) * (r2 x k*r)
[A2_x1,B2_x1,r2_x1] = interp_decomp(x1_core_reshape,tol);
x1_core_x = reshape(A2_x1,[r k r2_x1]);
x1_core_y = reshape(B2_x1',[r2_x1,k,r_tilde]);

% right boundary cores
right_boundary = reshape(A_tilde', [r_tilde k^2 1]);
right_boundary = reshape(right_boundary, [r_tilde*k k*1]);
% (r*k x r_right) * (r_right x k*1)
[A_right,B_right,r_right] = interp_decomp(right_boundary,tol);
core_right1 = reshape(A_right, [r_tilde k r_right]);
core_right2 = reshape(B_right', [r_right k 1]);

% form tensor train + 2 variables
core_cells = {   core_left1;    core_left2; ...
              common_core_x; common_core_y; ...
              common_core_x; common_core_y; ...
                  x1_core_x;     x1_core_y; ...
                core_right1;   core_right2};

svd_tt = cell2core(tt_tensor, core_cells);
% index out xA, xB
svd_tt = svd_tt(xB_ypos, xB_xpos, :, :, :, :, :, :, xA_ypos, xA_xpos);
norm(full(svd_tt)-exact_tt(:))

%% Test 4, R^2 points with indexing
clc; clear;
rng('default');

% Test 1: (2d, N = 3) exact tensor vs SVD
%       exp( -( |(xB - X_2)/dt - sin(X_2)|^2  + 
%               |(X_2 - xA)/dt - sin(xA)|^2 
k = 21;

% spatial grid in 2d, for each direction
dt = 1;
x_grid = linspace(-1,1,k);
[X_grid, Y_grid] = meshgrid(x_grid);

% fixed boundary points
xA = [-1,0];
xB = [1,0];

% for indexing
[xA_xpos, xA_ypos] = find_idx2d(xA, x_grid);
[xB_xpos, xB_ypos] = find_idx2d(xB, x_grid);


% flattened column-major
%[~, ~, ~, row_coord, col_coord] = flatten2d(x_grid);
all_points = [X_grid(:), Y_grid(:)];
% create symbolic var for exact computation
X = sym('x', [3,2]);
func_sym = exp(-( sum( ( ( xB-X(2,:) )/dt - sin( X(2,:) ) ).^2 ) + ...
           sum( ( ( X(2,:)-xA )/dt - sin( xA ) ).^2 ) ));
% convert to callable function
func_sym = matlabFunction(func_sym);

% build exact tensor
run_exact = true; % save time for quick runs
if run_exact
    exact_tt = zeros(k,k);
    for i1 = 1:k
        x2_1 = x_grid(i1);
        for i2 = 1:k 
            x2_2 = x_grid(i2);
                exact_tt(i2,i1) = ...
                    func_sym(x2_1,x2_2);
                %exact_tt(i1,i2,i3,i4,i5,i6) = ...
                %    func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
        end
    end
end

% build SVD tensor train
x_all = all_points(:,1); 
y_all = all_points(:,2);
% (1/dt^2) * |X_i - ( X_i-1 + dt * sin(X_i-1) )|^2
% build kernel
K_x = x_all' - (x_all + dt * sin(x_all));
K_y = y_all' - (y_all + dt * sin(y_all));

%K_x = x_all - (x_all + dt * sin(x_all))';
%K_y = y_all - (y_all + dt * sin(y_all))';
K = K_x.^2 + K_y.^2;
K = exp(-(1/dt^2) * K);

% first SVD to separate X1, X2, X3
tol = 1e-16;
[A,B,r] = interp_decomp(K,tol);
    
% (k^2 x r) * (r x k^2) 
common_core = zeros(r,k^2,r);
for i = 1:r
    for j = 1:r
        common_core(i,:,j) = A(:,i).*B(:,j);
    end
end

% second SVD to decouple x, y coords
common_core_reshape = reshape(common_core, [r*k, k*r]);
% (r*k x r2) * (r2 x k*r)
[A2,B2,r2] = interp_decomp(common_core_reshape,tol);
common_core_x = reshape(A2,[r k r2]);
common_core_y = reshape(B2',[r2,k,r]);
% boundary cores
left_boundary = reshape(B, [1 k^2 r]); % (1 x k^2 x r)
left_boundary = reshape(left_boundary, [1*k k*r]);
% (k x r_left) * (r_left x k*r)
[A_left,B_left,r_left] = interp_decomp(left_boundary,tol);
core_left1 = reshape(A_left, [1 k r_left]);
core_left2 = reshape(B_left', [r_left k r]);
    
right_boundary = reshape(A', [r k^2 1]); % (r x k^2 x 1)
right_boundary = reshape(right_boundary, [r*k k*1]);
% (r*k x r_right) * (r_right x k*1)
[A_right,B_right,r_right] = interp_decomp(right_boundary,tol);
core_right1 = reshape(A_right, [r k r_right]);
core_right2 = reshape(B_right', [r_right k 1]);
% form tensor train
core_cells = {core_left1; core_left2; ...
              common_core_x; common_core_y; ...
              core_right1; core_right2};

svd_tt = cell2core(tt_tensor, core_cells);

% index out xA, xB
svd_tt = svd_tt(xB_ypos,xB_xpos,:,:,xA_ypos,xA_xpos)
disp("> Norm error between ( Exact ) and ( SVD ), should expect macheps. \n")
norm(full(svd_tt)-exact_tt(:))

%% Test 6: Tests approx transition kernel works correctly
clc; clear;
rng('default');

% Test: (2d, N = 3) exact tensor vs SVD
% using the transition path kernel

k = 13;

% spatial grid in 2d, for each direction
dt = 0.01;
c = 2.5;
beta = 0.01;
x_grid = linspace(-2,2,k);
[X_grid, Y_grid] = meshgrid(x_grid);

% fixed boundary points
xA = [-1,0];
xB = [1,0];

% for indexing
[xA_xpos, xA_ypos] = find_idx2d(xA, x_grid);
[xB_xpos, xB_ypos] = find_idx2d(xB, x_grid);


% flattened column-major
%[~, ~, ~, row_coord, col_coord] = flatten2d(x_grid);
all_points = [X_grid(:), Y_grid(:)];
% create symbolic var for exact computation
X = sym('x', [3,2]);
func_sym = exp(-0.25 * energy_potential(X(:),dt,beta));
% convert to callable function
func_sym = matlabFunction(func_sym);

% build exact tensor
run_exact = true; % save time for quick runs
if run_exact
    exact_tt = zeros(k,k,k,k,k,k);
    for i1 = 1:k
        i1
        x1_1 = x_grid(i1);
        for i2 = 1:k 
            x1_2 = x_grid(i2);
            for i3 = 1:k
                x2_1 = x_grid(i3);
                for i4 = 1:k
                    x2_2 = x_grid(i4);
                    for i5 = 1:k
                        x3_1 = x_grid(i5);
                        for i6 = 1:k
                            x3_2 = x_grid(i6);
                            exact_tt(i6,i5,i4,i3,i2,i1) = ...
                                func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                            %exact_tt(i1,i2,i3,i4,i5,i6) = ...
                            %    func_sym(x1_1,x1_2,x2_1,x2_2,x3_1,x3_2);
                        end
                    end
                end
            end
        end
    end
end

% build SVD TT
svd_tt = approx_transition_kernel(x_grid, 3, 1e-10, dt, beta);
norm(full(svd_tt))
norm(exact_tt(:))
norm(full(svd_tt)-exact_tt(:))
%% Use approx_transition_kernel to sample
clear; clc; rng("default");
format long

% parameters (need to fine tune beta and dt)
beta = 1;
t_start = 0; t_end = 2;
N = 100; % tune N for finer solutions, effective dimension 2N

temperature = 1/beta;
dt = (t_end-t_start)/N;

% boundary conditions
xA = [-1; 0]; xB = [1; 0]; 

% need to discretize 2d grid for the path
L = 2;  k = 41;%21, 37;

% assume uniform grid
X_arr = linspace(-L,L,k);

% compress MATLAB array
X_tt = tt_tensor(X_arr);

% TT meshgrid
%X = tt_meshgrid_vert(X_tt, N*2); % effective dimension 2N
% TT meshgrid, variable ordering (x1,x2,...,xN,y1,y2,...,yN)
X = tt_meshgrid_vert(X_tt,2*N);

% SVD TT PDF (formal)
svd_path_pdf_tt = approx_transition_kernel(X_arr, N, 1e-8, dt, beta);
norm(svd_path_pdf_tt)
% generate uniform seed
M = 2^8; % number of sample points
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

[path_samples, path_sample_densities] = tt_irt_lin(gen_grid_arr, ...
    svd_path_pdf_tt, unif_sample);
%[path_samples, ...
%    path_sample_densities] = tt_irt_debias(M, log_path_pdf_func, ...
%    svd_path_pdf_tt, ...
%    gen_grid_arr, 'mcmc', true);

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
%% Plot Conditional Distribution from SVD Tensor Train
% run previous section to generate SVD TT
assert(exist("svd_path_pdf_tt"), ...
    "> Run previous section to generate SVD Tensor Train. ");












