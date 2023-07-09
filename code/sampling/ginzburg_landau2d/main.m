%% test sampling on Ginzburg Landau 2d

%   Complete sampling code from GL 2d Boltzmann distribution.

clear; clc; rng("default");
M = 10000; % number of ultrafine grid points
p = 30;   % highest Legendre polynomial order
Ns = 20000; % number of samples
% precomputations
n = p+1; % external bond dimension 
L = 1; % form ultrafine grid
xg = linspace(-L,L,M);
dx = xg(2)-xg(1);
xg = xg+dx/2;
xg(end) = [];
% ===== added
M = length(xg);
% create A matrix
A = get_legendre(xg,p,true)';

% generate coefficient train
d = 8;  % dimension of 2d grid (dimension for the dist. will be squared)
legendre_ord = p + 1;
% test d-dimensional Gaussian
x_sym = sym('x',[1,d]);
R = 2;
delta = 0.04;

% Ginzburg-Landau Boltzmann dist.
u_sym = sym('u', [d, d], 'real'); % symbolic 2d U grid
u_compare = u_sym;
% snake ordering, loop over columns, flip each even column
snake = false;
if snake
    for i = 1:d
        if mod(i,2) == 0
            u_sym(:,i) = flip(u_sym(:,i));
        end
    end
end
gl_func = ginzburg_landau_energy2d(R*u_sym(:), delta, d, snake);
% multi-dim PDF
%temp = 5;  % Ginzburg Landau paper
temp = 1.0; % (12/24/2022) added for parallel tempering comparison
beta = 1/temp;
equi_density = sqrt(exp(-beta*gl_func));
% convert to function
u_sym_temp = u_sym(:);
u_sym_temp = u_sym_temp';
f_ = matlabFunction(equi_density, 'Vars', {u_sym_temp});

[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, n, d^2, f_, legendre_ord, 1e-6);
if ~isfile("./gl2d_successful.mat")
    % if tensor file doesn't already exist, save
    save("./gl2d_successful.mat");
end
%%
if isfile("./gl2d_successful.mat")
    % if tensor file doesn't already exist, save
    clear; clc; rng("default");
    load("./gl2d_successful.mat");
end
C = coeff_tt;
C = C./norm(C);
%%
% truncate the coefficient train
max_rank = 3;
C = round(C, 1e-10, max_rank);
C = C/norm(C);
%%
% continuous tensor train samples
X = zeros(d^2,Ns);
for s=1:Ns % this could be parfor
    s
    X(:,s) = get_sample_copy(C,A,xg);
end
%%
% for each sample, reshape to a surface
X_2d = reshape(X,[d,d,Ns]);
% rescale back
X_2d = R.*X_2d;
%%
% plot samples as surface plots
X_2d_plot = zeros([d+2,d+2,Ns]);
X_2d_plot(2:d+1,2:d+1,:) = X_2d;
% boundary conditions on unit square 
% u|x=0,1 = 1, u|y=0,1 = -1
X_2d_plot(1,:,:) = 1; 
X_2d_plot(d+2,:,:) = 1;
X_2d_plot(:,1,:) = -1; 
X_2d_plot(:,d+2,:) = -1;
[Y_mesh,X_mesh] = meshgrid(linspace(0,1,d+2));
for k = 1:Ns
    figure(1); 
    contourf(X_mesh, Y_mesh, squeeze(X_2d_plot(:, :, k)), 'EdgeColor', 'red'); 
    zlim([-2,2]);
    shading interp; colorbar; pause(0.7);
end
%%
% plot histogram of marginals
for i = 1:d^2
    figure(1);
    histogram(R*X(i,:), 200); pause(0.5);
    title(strcat(['d = ', num2str(i)]))
end

figure(3);
% plot histogram of mean
histogram(R*mean(X,1), 200, 'FaceColor', 'red');

%%
% check to make sure norm is 1 before evaluating likelihood
norm(C)
likes = ftt_eval2(X,C);
% pick first half as training, second half as test
X_train = X(:,1:floor(0.5*Ns));
X_test = X(:,floor(0.5*Ns)+1:end);
likes_train = likes(1:floor(0.5*Ns));
likes_test = likes(floor(0.5*Ns)+1:end);

% generate Gaussian training data as comparison
% get exact covariance
exact_variance = true;
if exact_variance
    % rescale all samples in each dimension
    new_X = X;
    % generate Gaussian training data as comparison
    mu_samp = mean(new_X,2);
    tmp = repmat(mu_samp,1,Ns);
    cov_samp = (1/(Ns-1))*(new_X-tmp)*(new_X-tmp)';
else
    % use standard Gaussian
    mu_samp = 0.0 * mean(X,2);
    % make sure in [-1,1]
    cov_samp = 0.05 * eye(d^2);
end

X_bad = mvnrnd(mu_samp',cov_samp,Ns)';
likes_bad = mvnpdf(X_bad',mu_samp',cov_samp);
X_bad_train = X_bad(:,1:floor(0.5*Ns));
X_bad_test = X_bad(:,floor(0.5*Ns)+1:end);
likes_bad_train = likes_bad(1:floor(0.5*Ns));
likes_bad_test = likes_bad(floor(0.5*Ns)+1:end);

%%
% save the samples and exact likelihoods for normalizing flow processing
save("./data/legendre_tt_irt_ginzburg2d_sample.mat");