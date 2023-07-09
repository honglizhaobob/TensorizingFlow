% Follow example in Adaptive Monte Carlo augmented with normalizing Flows
% nonequilibrium path sampling using Functional-TT

% Modified energy potential
clear; clc; format long;
rng("default");

% parameter choices from [Gabrie]

% specify beta
beta = 0.5;

temperature = 1/beta;
fprintf(">> Temperature = %f\n\n", temperature);

t_start = 0; t_end = 1e-1;
N = 20; % tune N for finer solutions, effective dimension 2N
        % 50, extremely slow

dt = (t_end-t_start)/N;


% boundary conditions
xA = [-1,(0-5/6)]./3; xB = [1,(0-5/6)]./3; 


% ====================
%  FTT sampling setup
% ====================
M = 10000; % number of ultrafine grid points
p = 80;   % highest Legendre polynomial order
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

% test energy of a likely path
likely_path = zeros(N,2);
likely_path(:,1) = linspace(-1/3,1/3,N);
%disp(energy_potential_brownian_bridge(R*likely_path(:), dt, beta))
disp(energy_potential2(R*likely_path(:), dt, beta))

%%
% create function handle
U = sym('u', [N, 2], 'real');
%path_action = energy_potential_brownian_bridge2(R*U(:), dt, beta);
path_action = energy_potential2(R*U(:), dt, beta);

% sqrt PDF
equi_density = sqrt(exp(-path_action));
u_sym = U(:)';
% convert to function
f_ = matlabFunction(equi_density, 'Vars', {u_sym});
[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, k, d, f_, legendre_ord, 1e-8);
C = coeff_tt;
%%
% truncate the coefficient train
max_rank = 30;
C = round(C, 1e-10, max_rank);
%%
load("noneq_tpt_100cross_rank395.mat");
Ns=300;
C = C/norm(C);
% continuous tensor train sample

X = zeros(d,Ns);
exact_probs = zeros(Ns,1);
for s=1:Ns % this could be parfor
    s
    X(:,s) = get_sample_copy(C,A,xg);
    % evaluate exact probability
    exact_probs(s) = exp(-energy_potential2(X(:,s),dt,beta));
end
close;


%%

% clean data for TT approximation error
all_prune_idx = [];
for i = 1:size(X,2)
    sample_path = X(:,i);
    if any(abs(R*sample_path)>=2)
        all_prune_idx = [i all_prune_idx];
    end
end

% prune unlikely paths
save_X = X;
save_X(:,all_prune_idx) = [];



%%
% Use the following code to plot
for i = 1:size(X,2)
    sample_path = X(:,i);
    sample_path = R*reshape(sample_path', [N 2]);
    sample_path = [xA; sample_path; xB];
    figure(1); hold on;
    plot(sample_path(:,1), sample_path(:,2),'LineWidth',2);
end


%% Likelihood evaluation from TT

prior_logpdf = zeros(size(X,2),1);
% evaluate log-likelihood from Brownian bridge
for i = 1:size(X,2)
    % get sample path
    sample_path = R*X(:,i);
    % evaluate log-pdf (not normalized)
    prior_logpdf(i) ...
        = -energy_potential_brownian_bridge(sample_path,dt,beta);
end

%% Train-test split
% evaluate likelihood
%likes = ftt_eval2(X,C);

% directly use exact likelihood
likes = exp(prior_logpdf);
%%
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


