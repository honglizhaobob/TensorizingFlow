% Script for sampling from the anti-ferromagnetic 
% GL 1d distribution. (spin-glass) with 
% uniformly randomly generated scaling on [-1, 1]

% Date: 01/31/2022
% Author: Hongli Zhao, honglizhaobob@uchicago.edu

%% test sampling on Ginzburg Landau 1d
%   Complete main driver code for sampling from GL 1d Boltzmann
% distribution.

% working examples:
%
%   p = 30, R = 2, temp = 8, delta = 0.04, max_rank = 3, k = 80, d = 50
clear; clc; rng(2,"philox");
M = 10000; % number of ultrafine grid points
p = 50;   % highest Legendre polynomial order
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
d = 35; % dimension of multivariate distribution; spin glass 35

      % d = 30 (spin glass)
k = 100; % number of grid points in each dimension
legendre_ord = p + 1;

% Ginzburg-Landau Boltzmann dist.

% domain size
R = 3.5; % spin glass: 3.5, anti: 4.5
% parameters
delta = 0.04; 
temp = 16; % the choices come from TT committor paper (Yian et al.)
beta = 1/temp;

% create function handle
u_sym = sym('u', [1, d]); % symbolic U1, U2, ..., Ud

% generate scaling
spin_glass_scaling = unifrnd(-1,1,[d-1,1]); % coeff for (U2-U1)^2, (U3-U2)^2 ... (Ud-Ud-1)^2

gl_func = ginzburg_landau_anti_energy1d(R*u_sym, delta, spin_glass_scaling);

% sqrt(pi(x))
equi_density = sqrt(exp(-beta*gl_func));
% convert to function
f_ = matlabFunction(equi_density, 'Vars', {u_sym});
[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, k, d, f_, legendre_ord, 1e-9);
C = coeff_tt;
%%
% truncate the coefficient train
max_rank = 1;
C = round(coeff_tt, 1e-10, max_rank);
%%
C = C/norm(C);
% continuous tensor train samples
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    s
    X(:,s) = get_sample_copy(C,A,xg);
end

%%
% Plotting for sanity check
% plot histogram of marginals
for i = 1:d
    figure('visible','off');
    histogram(X(i,:), 200); pause(0.5);
    title("Histogram of Discretized Scalar Field under Ginzburg-Landau Energy Potential (1D)");
    xlabel("sample $u(x)$", 'interpreter', 'latex')
    ylabel("count", 'interpreter', 'latex')
    % save figure
    %saveas(gcf,strcat(['./img/GL_1d/hist/gl1d_d', num2str(i), '.png']))
end
%%
fig = figure('PaperUnits', 'inches', 'PaperSize', ...
    [8 5], 'PaperPosition', [0 0 8 5]);

% plot histogram of mean
histogram(R*sum(X,1), 300, 'FaceColor',...
    'red', 'Normalization','probability');
%title("Histogram of $u_+$, $u_-$",'interpreter', 'latex');
xlabel("Sum of States", 'interpreter', 'latex', 'FontSize',20);
ylabel("Density", 'interpreter', 'latex', 'FontSize',20);
hold on;
% fit kde
[f,xi] = ksdensity(R*sum(X,1));
plot(xi,f/25,'LineStyle','--','LineWidth',2.5,'Color','black');
ax = gca;
ax.FontSize = 16;
%saveas(fig, strcat(['./img/gl1d_d_mean', '.png']))
exportgraphics(fig,'./img/gl1d_d_mean.png','Resolution',1000)

if ~isfile("./gl1d_anti_full_rank.mat")
    % save file if it is not here
    save("./gl1d_anti_full_rank.mat");
end
%% compute likelihoods from continuous TT (GL 1d)
if isfile("./gl1d_anti_full_rank.mat")
    % if data already exists, load
    clear; clc; rng("default");
    load("./gl1d_anti_full_rank.mat");
end
%%
% evaluate likelihood
likes = ftt_eval2(X,C);
X_save = X;
% pick first half as training, second half as test
X_train = X_save(:,1:floor(0.5*Ns));
X_test = X_save(:,floor(0.5*Ns)+1:end);
likes_train = likes(1:floor(0.5*Ns));
likes_test = likes(floor(0.5*Ns)+1:end);

exact_variance = false;
if exact_variance
    % rescale all samples in each dimension
    new_X = X;
    % generate Gaussian training data as comparison
    mu_samp = mean(new_X,2);
    tmp = repmat(mu_samp,1,Ns);
    cov_samp = (1/(Ns-1))*(new_X-tmp)*(new_X-tmp)';
else
    % use standard Gaussian
    mu_samp = 0 * mean(X,2);
    % make sure in [-1,1]
    cov_samp = 0.05 * eye(d);
end

% generate Gaussian training data as comparison
X_bad = mvnrnd(mu_samp',cov_samp,Ns)';
likes_bad = mvnpdf(X_bad',mu_samp',cov_samp);
X_bad_train = X_bad(:,1:floor(0.5*Ns));
X_bad_test = X_bad(:,floor(0.5*Ns)+1:end);
likes_bad_train = likes_bad(1:floor(0.5*Ns));
likes_bad_test = likes_bad(floor(0.5*Ns)+1:end);
