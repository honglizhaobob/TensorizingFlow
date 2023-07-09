% Hongli Zhao, honglizhaobob@uchicago.edu
% Main script for tensor train sampling from
% a 2D multimodal mixture Gaussian. To check
% if KL divergence of normalizing flow gets 
% stuck in local minima as hypothesized.
clear; clc; rng('default');

% reference: https://www.mathworks.com/help/stats/simulate-data-from-a-gaussian-mixture-model.html
% mean and covmats (8 modes)

% highest number of polynomials
p = 512;
legendre_ord = p + 1; 
% generate metadata for grid
d = 2;
inp = zeros(d,3);
for i = 1:d
    inp(i,1) = -1;
    inp(i,2) = 1;
    inp(i,3) = 2^10;   
end

% 8 modes, specify in two dimensions
mu1 = [2,0];
mu2 = [-2,0];
mu3 = [0,2];
mu4 = [0,-2];
mu5 = [sqrt(2),sqrt(2)];
mu6 = [sqrt(2),-sqrt(2)];
mu7 = [-sqrt(2),sqrt(2)];
mu8 = [-sqrt(2),-sqrt(2)];


% covariance matrices




covmat1 = 0.05*eye(d);
covmat2 = 0.05*eye(d);
covmat3 = 0.05*eye(d);
covmat4 = 0.05*eye(d);
covmat5 = 0.05*eye(d);
covmat6 = 0.05*eye(d);
covmat7 = 0.05*eye(d);
covmat8 = 0.05*eye(d);



% create mixture Gaussian PDF with 5 modes
sym_theta = sym('theta', [1,d]);

scaling = 3;
% use symbolic computation
raw_func = 0;
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu1,covmat1);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu2,covmat2);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu3,covmat3);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu4,covmat4);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu5,covmat5);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu6,covmat6);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu7,covmat7);
raw_func = raw_func + mvnpdf(scaling*sym_theta,mu8,covmat8);

% sqrt of density
raw_func = sqrt(raw_func);

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% test sampling
tol = 1e-10; % set to 1e-10 for accuracy
[coeff_tt, ~] = ...
    legendre_ftt_cross(inp, r_func, legendre_ord, tol);

C = coeff_tt;

% number of samples
Ns = 20000;

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

%% Visualize marginal

% use same number of legendre quadrature
% to evaluate and integrate out last 2 marginals
legendre_grids = cell(d,1);
legendre_weights = cell(d,1);
legendre_data = cell(d,1);
for i = 1:d
    [nodes,W] = lgwt(inp(i,3),inp(i,1),inp(i,2));
    legendre_grids{i} = nodes;
    legendre_weights{i} = W;
    % get legendre polynomial data 
    % on legendre grids
    legendre_data{i} = get_legendre(nodes,p,true)';
end

% contract to get full joint
coeff_cores = core(C);
density_cores = cell(d,1);
density_cores{1} = tns_mult(coeff_cores{1},1,legendre_data{1},1);
density_cores{2} = tns_mult(coeff_cores{2},1,legendre_data{2},1);

density_cores{1} = permute(density_cores{1},[2 1]);
density_cores{1} = reshape(density_cores{1},...
    [1,size(density_cores{1})]);
% create full density
full_joint = cell2core(tt_tensor(),density_cores);
% read matrix
full_joint = full(full_joint,[2^10, 2^10]);
% visualize
[Y,X] = meshgrid(legendre_grids{1},legendre_grids{2});
figure(1);
surf(X,Y,full_joint); shading interp;
view(2);
%% truncate the coefficient train
truncate = true;
if truncate
    % perform rank truncation
    max_rank = 2;
    C = round(coeff_tt, 1e-10, max_rank);
end

C = C./norm(C);

%% sample from mixture
C = C./norm(C);
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    s
    X(:,s) = get_sample(C,all_A,all_xg);
end
save_X = X;
%% Plot samples
figure(1);
scatter(X(1,:), X(2,:), 5, 'blue');
grid on;
%% 
% get exact covariance
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

%% 
% evaluate exact likelihood
likes = ftt_eval3(X,C);
% pick first half as training, second half as test
X_train = X(:,1:floor(0.5*Ns));
X_test = X(:,floor(0.5*Ns)+1:end);
likes_train = likes(1:floor(0.5*Ns));
likes_test = likes(floor(0.5*Ns)+1:end);

X_bad = mvnrnd(mu_samp',cov_samp,Ns)';
likes_bad = mvnpdf(X_bad',mu_samp',cov_samp);
X_bad_train = X_bad(:,1:floor(0.5*Ns));
X_bad_test = X_bad(:,floor(0.5*Ns)+1:end);
likes_bad_train = likes_bad(1:floor(0.5*Ns));
likes_bad_test = likes_bad(floor(0.5*Ns)+1:end);

%% save data
save("mixture_gaussian2d_full_rank.mat");



