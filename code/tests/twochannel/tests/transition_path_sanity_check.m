% Various tests for understanding path action.
%% plot potential contour
clear; clc; rng('default');
k = 100; L = 1;
x_start = -L; x_end = L;
xgrid = linspace(x_start, x_end, k);
[X,Y] = meshgrid(xgrid, xgrid);
% column-wise flatten
all_points = [X(:) Y(:)];

% amplitudes for the four gaussians
% original params (Metzner): 
A1 = 3; A2 = -3; A3 = -4; A4 = -4;
A5 = 0;

% means for the four gaussians
mu1 = [0,(1/3-5/6)]./3; mu2 = [0,(5/3-5/6)]./3; 
mu3 = [-1,(0-5/6)]./3;  mu4 = [1,(0-5/6)]./3;
[~, ~, f1] = gaussian_function2(all_points, A1, mu1);
[~, ~, f2] = gaussian_function2(all_points, A2, mu2);
[~, ~, f3] = gaussian_function2(all_points, A3, mu3);
[~, ~, f4] = gaussian_function2(all_points, A4, mu4);
Z = f1 + f2 + f3 + f4;

% following original equation in TPT
Z = Z + A5*all_points(:,1).^4 + A5*( all_points(:,2) - 1/3 ).^4;
Z_grid = reshape(Z, [k k]);

% energy contour
figure(1);
sgtitle("Energy Contour Plot of the Three-Hole Potential");
%contourf(X,Y,Z_grid,35); colorbar;
surfc(X,Y,Z_grid); colorbar; shading interp;

%% Create connecting paths and plotting energy
clear; clc; rng('default');
xA = [-1,(0-5/6)]./3;
xB = [1,(0-5/6)]./3;
xUp = [0,(5/3-5/6)]./3;
xDown=[0,-0.43];

% direct path
beta = 0.5;   % 6.67
all_N = 5:5:1e+4;
all_energy1 = zeros(length(all_N),1);
all_energy2 = zeros(length(all_N),1);
all_energy3 = zeros(length(all_N),1);
t_end = 1;
for i = 1:length(all_N)
    i
    N = all_N(i);
    dt = 1/N;
    path1 = zeros(N,2);

    % direct path [-1,0]-->[0,-1/3]-->[1,0]
    path1(1:floor(N/2),1)=linspace(-1/3,0,floor(N/2));
    path1(floor(N/2)+1:end,1)=linspace(0,1/3,N-floor(N/2));
    path1(1:floor(N/2),2)=linspace(xA(2),xDown(2),floor(N/2));
    path1(floor(N/2)+1:end,2)=linspace(xDown(2),xA(2),N-floor(N/2));

    % direct path [-1,0]-->[1,0]
    path2 = zeros(N,2);
    path2(:,1) = linspace(-1/3,1/3,N);
    path2(:,2) = (0-5/6)/3;
  

    % direct path [-1,0]-->[0,5/3]-->[1,0]
    path3 = zeros(N,2);
    path3(1:floor(N/2),1)=linspace(-1/3,0,floor(N/2));
    path3(floor(N/2)+1:end,1)=linspace(0,1/3,N-floor(N/2));
    path3(1:floor(N/2),2)=linspace(xA(2),xUp(2),floor(N/2));
    path3(floor(N/2)+1:end,2)=linspace(xUp(2),xA(2),N-floor(N/2));


    % plot
    %figure(1);
    %plot(path1(:,1),path1(:,2),'LineWidth',2.5,'Color','green');
    %xlim([-1,1]); ylim([-1,1]);
    % compute path energy

    % delete boundary points since energy function already has it
    path1(1,:) = []; path1(end,:) = [];
    all_energy1(i)=energy_potential2(path1,dt,beta);
    all_energy2(i)=energy_potential2(path2,dt,beta);
    all_energy3(i)=energy_potential2(path3,dt,beta);
end

figure(1);
a1=plot(all_N,all_energy1,'LineWidth',2.5,'Color','blue'); hold on;

%plot(all_N,all_energy2,'LineWidth',2.5,'Color','red'); grid on;
a2=plot(all_N,all_energy3,'LineWidth',2.5,'Color','black'); 
legend('Lower');


%% Generating random points and evaluating 
% randomly sample 2d points in [-1,1]^2, treating them as a path;
% evaluate the exact path probability with varying N, beta.
clear; clc; rng('default');
all_N = 10:10:120;
all_temp = [0.01, 0.05, 0.1, 0.2, ...
    0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
t_end = 1;
n_samples = 500; % number of random paths
all_random_path_samples = cell(length(all_N),length(all_temp));
% compute all path energy
all_path_energy = zeros(length(all_N),length(all_temp),n_samples);
for i = 1:length(all_N)
    i
    for j = 1:length(all_temp)
        N = all_N(i);
        temp = all_temp(j);
        beta = 1/temp;
        dt = t_end/N;
        % randomly sample points in [-1,1]^2
        all_samples = unifrnd(-1,1,[n_samples,2*N]);
        % store
        all_random_path_samples{i,j} = all_samples;
        % evaluate all path probabilities
        path_energy = zeros(n_samples,1);
        for k = 1:n_samples
            sample_path = all_samples(k,:)';
            all_path_energy(i,j,k) = ...
                energy_potential2(sample_path,dt,beta);
        end
    end
end

%%
% evaluate all relative probabilities
% Relative Prob. = exp(-E1)/exp(-Emax) = exp(Emax-E1)
all_path_relative_prob = zeros(length(all_N),length(all_temp),...
    n_samples);
for i = 1:length(all_N)
    i
    for j = 1:length(all_temp)
        % get all sample path energy
        sample_energy = all_path_energy(i,j,:);
        sample_energy = sample_energy(:);
        % take min of energy --> max of probability 
        Emax = min(sample_energy);
        % evaluate relative probability
        all_path_relative_prob(i,j,:) = ...
            exp(Emax-sample_energy);
    end
end

%% 
% create movie
vidfile = VideoWriter('histogram.mp4','MPEG-4');
open(vidfile);

% visualize relative probabilities with historgram
for i = 1:length(all_N)
    i
    for j = 1:length(all_temp)
        % get all sample path relative probability
        sample_rel_prob = squeeze(all_path_relative_prob(i,j,:));
        h1 = figure(1);
        histogram(sample_rel_prob,80);
        im = getframe(h1);
        sample_rel_prob(sample_rel_prob==1)=[];    % get rid of max path itself
        title(sprintf("$T = %f, N = %d$",...
            all_temp(j),all_N(i)), ...
            'Interpreter','latex');
        subtitle("Histogram of Random Path Relative Probability Distribution ");
        pause(0.3);
        xlim([0,1]); ylim([0,n_samples]);
        writeVideo(vidfile, im);
    end
end

close(vidfile);

% ========================================================================
%% Use AMEN_CROSS to sample one path
% ========================================================================
% create energy contour with 1 path
clear; clc; rng('default');
kx = 100;
ky = 100;
x_start = -0.4; x_end = 0.4;
%x_start = -1; x_end = 1;
y_start = -0.2; y_end = -0.36;
%y_start = -1; y_end = 1;
xgrid = linspace(x_start, x_end, kx);
ygrid = linspace(y_start, y_end, ky);
[X,Y] = meshgrid(xgrid, ygrid);
% column-wise flatten
all_points = [X(:) Y(:)];

% amplitudes for the four gaussians
% original params (Metzner): 
A3 = -0.2; A4 = -0.2;
A5 = 0;

% means for two gaussians
mu3 = [-1,(0-5/6)]./3;  mu4 = [1,(0-5/6)]./3;

[~, ~, f3] = gaussian_function2b(all_points, A3, mu3,2.25);
[~, ~, f4] = gaussian_function2b(all_points, A4, mu4,2.25);
Z = f3 + f4;

% following original equation in TPT
Z = Z + A5*all_points(:,1).^4 + A5*( all_points(:,2) - 1/3 ).^4;
Z_grid = reshape(Z, [kx ky]);

% energy contour
figure(1);
sgtitle("Energy Contour Plot of the Two-Hole Potential");
%contourf(X,Y,Z_grid,35); colorbar;
surfc(X,Y,Z_grid); colorbar; shading interp;

%%
% Verify amen_cross, do discrete sampling
beta = 6.67;
N = 20;
t_end = 1;
dt = t_end/N;
% create TT grid
tt_grid_cells = cell(2*N,1);
for i=1:N
    tt_grid_cells{i} = tt_tensor(xgrid);
end
for i=N+1:2*N
    tt_grid_cells{i} = tt_tensor(ygrid);
end
% TT meshgrid
X = tt_meshgrid_vert(tt_grid_cells);
sym_x = sym("x", [N,2], 'real');
% pdf
tpt_pdf = exp(-two_hole_energy_flat(sym_x,dt,beta));
%tpt_pdf = exp(-flat_energy(sym_x,dt,beta));    % V is flat surface
% create function handle
sym_x = sym_x(:)';
tpt_pdf = matlabFunction(tpt_pdf, 'Vars', {sym_x});
ftt = amen_cross_s(X,tpt_pdf,1e-9,'nswp',50);

% grid for sampling 
sampling_grid_cells = cell(2*N,1);
for i=1:N
    sampling_grid_cells{i} = xgrid';
end
for i=N+1:2*N
    sampling_grid_cells{i} = ygrid';
end

num_samples = 5000;
unif_sample = rand(num_samples,length(sampling_grid_cells));
[xq, lFapp] = tt_irt_lin(sampling_grid_cells, ftt, unif_sample);
for i = 1:num_samples
    sample_path = reshape(xq(i,:),[N,2]);
    figure(1); hold on;
    plot(sample_path(:,1),sample_path(:,2),'LineWidth',2.5);
end








% ========================================================================
%% Create paths and plotting distribution (???)
% ========================================================================

% start from [-1,(0-5/6)]./3, create a grid of destinations
% and evaluate energy of N points along a direct path
% from starting point to destination. This energy is used 
% for computing probability.
clear; clc; rng('default');
xA = [-1,(0-5/6)]./3;

% create a grid of xB (destination)
sqrt_m = 200;
xgrid = linspace(-1,1,sqrt_m);
ygrid = linspace(-1,1,sqrt_m);
[xgrid,ygrid] = meshgrid(xgrid,ygrid);
xBs = [xgrid(:),ygrid(:)];
m = size(xBs,1);
N = 100;    % discretization level

all_path_energy = zeros(m,1);

% Physical parameters
beta = 0.01;
dt = 1/N;

for i=1:m
    i
    x1=xA(1); y1=xA(2);
    xB = xBs(i,:);
    % create direct path connecting xA=(x1,y1), xB=(x2,y2)
    x2=xB(1); y2=xB(2);
    % reassign x1 x2 y1 y2 based on ordering
    flag=0;
    if x2<=x1
        tmp=x1;
        x1=x2;
        x2=tmp;
        tmp=y1;
        y1=y2;
        y2=tmp;
        flag=1;
    end

    fprintf("%f,%f",x1,x2)
    fprintf("%f,%f",y1,y2)
    slope = (y2-y1)/(x2-x1);
    intercept = y2-x2*slope;
    % create path
    path = zeros(N,2);
    path(:,1)=linspace(x1,x2,N);
    path(:,2)=slope*path(:,1)+intercept;
    if flag==1
        % reverse order
        path(:,1)=flip(path(:,1));
        path(:,2)=flip(path(:,2));
    end
    %figure(1);
    %plot(path(:,1),path(:,2),'LineWidth',2.4);
    %xlim([-1,1]); ylim([-1,1]);
    path = path';
    % evaluate 
    all_path_energy(i) = energy_potential2(path, dt, beta);
end

% reshape and plot path action surface
surfc(xgrid,ygrid,reshape(exp(-all_path_energy),sqrt(m),[]));
colorbar; shading interp;











%% test new energy potential
clear; clc; rng('default');

k = 100; L = 1;
x_start = -L; x_end = L;
xgrid = linspace(x_start, x_end, k);
[X,Y] = meshgrid(xgrid, xgrid);
% column-wise flatten
all_points = [X(:) Y(:)];

% create a series of Gaussians concentrated 
% on a path connecting [-1,0], [1,0], 
% so we create a valley.

% create a path with 200 points, 
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
f=0;
for i = 1:path_num
    [~,~,tmp]=gaussian_function2b(all_points, all_coeffs(i),...
        all_means(i,:),8);
    f=f+tmp;
end

% plot surface
Z_grid = reshape(f, [k k]);

% energy contour
figure(1);
sgtitle("Energy Contour Plot of the Three-Hole Potential");
%contourf(X,Y,Z_grid,35); colorbar;
surfc(X,Y,(Z_grid)); colorbar; shading interp;

%% Sampling engineered energy potential
% Path sampling of simple one-path valley

% Modified energy potential
clear; clc; format long;
rng("default");

% parameter choices from [Gabrie]

% specify beta
beta = 1;

temperature = 1/beta;
fprintf(">> Temperature = %f\n\n", temperature);

t_start = 0; t_end = 5e-1;
N = 10; % tune N for finer solutions, effective dimension 2N
        % 50, extremely slow

dt = (t_end-t_start)/N;


% boundary conditions
xA = [-0.7,-0.4];
xB = [0.7,-0.4];


% ====================
%  FTT sampling setup
% ====================
M = 10000; % number of ultrafine grid points
p = 50;   % highest Legendre polynomial order
Ns = 100; % number of samples
% precomputations

L = 1; % form ultrafine grid
xg = linspace(-L,L,M);
dx = xg(2)-xg(1);
xg = xg+dx/2;
xg(end) = [];
% ===== added
M = length(xg);
% create A matrix
A = get_legendre(xg,p,true)';

% Begin Sampling
d = 2*N; % dimension of multivariate distribution
k = p+1; % number of grid points in each dimension (recommended: p+1)
legendre_ord = p + 1;

% TPT distribution
% scaling (to [-1,1])
R = 1; % effective domain [-R,R]x[-R,R]


% create function handle
U = sym('u', [N, 2], 'real');
path_action = energy_potential3(R*U(:), dt, beta);

% sqrt PDF
equi_density = sqrt(exp(-path_action));
u_sym = U(:)';
% convert to function
f_ = matlabFunction(equi_density, 'Vars', {u_sym});
[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, k, d, f_, legendre_ord, 1e-6);
C = coeff_tt;

%%
C = C/norm(C);
% continuous tensor train sample

X = zeros(d,5*Ns);
for s=1:5*Ns % this could be parfor
    s
    X(:,s) = get_sample_copy(C,A,xg);
end
close;
% Use the following code to plot
for i = 1:size(X,2)
    sample_path = X(:,i);
    sample_path = R*reshape(sample_path', [N 2]);
    sample_path = [xA; sample_path; xB];
    figure(1); hold on;
    plot(sample_path(:,1), sample_path(:,2),'LineWidth',2);
end


%%

% helper functions
function B = two_hole_potential(x, c)
    % two-hole potential created for quick testing
    % of sampling one path

    % means
    mu3 = [-1,(0-5/6)]./3;  mu4 = [1,(0-5/6)]./3;

    % strengths of local min/maxima
    A3 = -4; A4 = -4; 
    A5 = 0;
    
    % drift part
    f = c * x(:, [2,1]);
    f(:,1) = -f(:,1);
    
    % get gradients from sum of Gaussian functions
    [grad3_dx, grad3_dy, ~] = gaussian_function2(x, A3, mu3);
    [grad4_dx, grad4_dy, ~] = gaussian_function2(x, A4, mu4);
    % gradient part 1
    grad3 = [grad3_dx, grad3_dy];
    grad4 = [grad4_dx, grad4_dy];
    
    % four Gaussians + additional penalty on mu1
    B = - (grad3 + grad4) + f;
end


function B = flat_potential(x, c)
    % a potential that is just a flat plane
    B = 0;
end

function B = two_hole_potential_flat(x, c)
    % two-hole potential created for quick testing
    % of sampling one path. But it's quite flat.

    % means
    mu3 = [-1,(0-5/6)]./3;  mu4 = [1,(0-5/6)]./3;

    % strengths of local min/maxima
    A3 = -0.2; A4 = -0.2; 
   
    
    % get gradients from sum of Gaussian functions
    [grad3_dx, grad3_dy, ~] = gaussian_function2b(x, A3, mu3,2.25);
    [grad4_dx, grad4_dy, ~] = gaussian_function2b(x, A4, mu4,2.25);
    % gradient part 1
    grad3 = [grad3_dx, grad3_dy];
    grad4 = [grad4_dx, grad4_dy];
    
    % four Gaussians + additional penalty on mu1
    B = - (grad3 + grad4);
end


function S = two_hole_energy(path, dt, beta)
    % Helper function for evaluating energy
    % of paths sampled from two-hole potential.
    
    X = reshape(path, [], 2);
    c = 0;
    % boundary points (shifted and rescaled)
    xA = [-1,(0-5/6)]./3; 
    xB = [1,(0-5/6)]./3;
    
    % number of nodes
    N = size(X, 1);
    S = 0;
    
    % use forward differencing
    % original: \int |dX/dt - b(x)|^2 dt 
    %           \approx dt * \sum_i |(X_i+1 - X_i)/dt - b(X_i)|^2

    if N > 1
        S_vec = (1/dt)*(X(2:N, :) - X(1:(N-1), :)) - ...
            two_hole_potential(X(1:(N-1), :), c);
        S = sum(sum(S_vec.^2));
    end

    % add penalty from boundary
    S = S + sum( ( (1/dt)*(X(1, :)-xA) - two_hole_potential(X(1,:),c) ).^2 );
    S = S + sum( ( (1/dt)*(xB-X(end, :)) - two_hole_potential(X(end,:),c) ).^2 );
    S = beta * dt * S;
end


function S = flat_energy(path, dt, beta)
    % Helper function for evaluating energy
    % of paths sampled from two-hole potential.
    
    X = reshape(path, [], 2);
    c = 0;
    % boundary points (shifted and rescaled)
    xA = [-1,(0-5/6)]./3; 
    xB = [1,(0-5/6)]./3;
    
    % number of nodes
    N = size(X, 1);
    S = 0;
    
    % use forward differencing
    % original: \int |dX/dt - b(x)|^2 dt 
    %           \approx dt * \sum_i |(X_i+1 - X_i)/dt - b(X_i)|^2

    if N > 1
        S_vec = (1/dt)*(X(2:N, :) - X(1:(N-1), :)) - ...
            flat_potential(X(1:(N-1), :), c);
        S = sum(sum(S_vec.^2));
    end

    % add penalty from boundary
    S = S + sum( ( (1/dt)*(X(1, :)-xA) - flat_potential(X(1,:),c) ).^2 );
    S = S + sum( ( (1/dt)*(xB-X(end, :)) - flat_potential(X(end,:),c) ).^2 );
    S = beta * dt * S;
end


function S = two_hole_energy_flat(path, dt, beta)
    % Helper function for evaluating energy
    % of paths sampled from two-hole potential.
    
    X = reshape(path, [], 2);
    c = 0;
    % boundary points (shifted and rescaled)
    xA = [-1,(0-5/6)]./3; 
    xB = [1,(0-5/6)]./3;
    
    % number of nodes
    N = size(X, 1);
    S = 0;
    
    % use forward differencing
    % original: \int |dX/dt - b(x)|^2 dt 
    %           \approx dt * \sum_i |(X_i+1 - X_i)/dt - b(X_i)|^2

    if N > 1
        S_vec = (1/dt)*(X(2:N, :) - X(1:(N-1), :)) - ...
            two_hole_potential_flat(X(1:(N-1), :), c);
        S = sum(sum(S_vec.^2));
    end

    % add penalty from boundary
    S = S + sum( ( (1/dt)*(X(1, :)-xA) - two_hole_potential_flat(X(1,:),c) ).^2 );
    S = S + sum( ( (1/dt)*(xB-X(end, :)) - two_hole_potential_flat(X(end,:),c) ).^2 );
    S = beta * dt * S;
end








