% Hongli Zhao, honglizhaobob@uchicago.edu
% Main script for tensor train sampling from
% a 3D double rosenbrock.
% if KL divergence of normalizing flow gets 
% stuck in local minima as hypothesized.
clear; clc; rng('default');
% highest number of polynomials
p = 512;
legendre_ord = p + 1; 
% generate metadata for grid
d = 3;

% grid points in each dimension
inp = zeros(d,3);
for i = 1:d
    if i < d-1
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^9;
    end
    if i == d-1
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^11;
    end
    if i == d
        inp(i,1) = -1;
        inp(i,2) = 1;
        inp(i,3) = 2^14;
    end      
end

% create Rosenbrock function handle for d-dimensions
sym_theta = sym('theta', [1,d]);
% use symbolic computation
raw_func = 0;
for k = 1:d-1
    if k <= d-3
        % rescale all dimensions by 2
        raw_func = raw_func + ((2*sym_theta(k))^2 + ...
            ((2*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-2
        % rescale (d-1) by 7, (d-2) by 2
        raw_func = raw_func + ((2*sym_theta(k))^2 + ...
            ((7*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-1
        % rescale d by 200, (d-1) by 7
        raw_func = raw_func + ((7*sym_theta(k))^2 + ...
            ((200*sym_theta(k+1)) + 5 * ((7*sym_theta(k))^2 + 1) )^2);
    end
end

raw_func = exp(-raw_func/2);

raw_func2 = 0;
% build second Rosenbrock (shifted)
for k = 1:d-1
    if k <= d-3
        % rescale all dimensions by 2
        raw_func2 = raw_func2 + ((2*sym_theta(k))^2 + ...
            ((2*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-2
        % rescale (d-1) by 7, (d-2) by 2
        raw_func2 = raw_func2 + ((2*sym_theta(k))^2 + ...
            ((7*sym_theta(k+1)) + 5 * ((2*sym_theta(k))^2 + 1) )^2);
    end
    if k == d-1
        % rescale d by 200, (d-1) by 7
        % flip the last dimension (d)
        raw_func2 = raw_func2 + ((7*sym_theta(k))^2 + ...
            ((200*(-sym_theta(k+1))) + 5 * ((7*sym_theta(k))^2 + 1) )^2);
    end
end

% sqrt of density
raw_func2 = exp(-raw_func2/2);

% combine the mixture
raw_func = 0.5 * raw_func + 0.5 * raw_func2;
raw_func = sqrt(raw_func);

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% test sampling
tol = 1e-11; % set to 1e-10 for accuracy
[coeff_tt, ~] = ...
    legendre_ftt_cross(inp, r_func, legendre_ord, tol);

%% Plot marginal
% check exists 
legendre_ord; d;
inp; p;

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
%coeff_cores = core(coeff_tt);
coeff_cores = core(C);
density_cores = cell(d,1);
for i = 1:d
    % contract with legendre data to get density cores
    density_cores{i} = tns_mult(coeff_cores{i},1,legendre_data{i},1);
end

% reshape cores appropriately
for i = 1:d
    if i == 1
        density_cores{i} = permute(density_cores{i},[2 1]);
        density_cores{i} = reshape(density_cores{i},...
            [1,size(density_cores{i})]);
    elseif i == d
        continue;
    else
        density_cores{i} = permute(density_cores{i},[1 3 2]);
    end
end


% create full density
full_joint = cell2core(tt_tensor(),density_cores);


marginalized = core(full_joint);
for i = 1:d
    % first reshape to appropriate shape
    if i == 1
        marginalized{i} = reshape(marginalized{i},...
            [1,size(marginalized{i})]);
    elseif i == d
        marginalized{i} = permute(marginalized{i},[2 1]);
    else
        marginalized{i} = permute(marginalized{i},[2 1 3]);
    end
end

% marginalize all except last 3 by contracting with weights
for i = 1:d-3
    marginalized{i} = tns_mult(marginalized{i},2,...
        legendre_weights{i},1);
end

last_three = marginalized{1};
for i = 2:d-2
    % contract with the next core
    last_three = tns_mult(last_three,2,marginalized{i},1);
end

marginalized_new = cell(3,1);
marginalized_new{1} = last_three;
marginalized_new{2} = marginalized{end-1};
marginalized_new{3} = marginalized{end};

save_marginalized_new = marginalized_new;

% (d-2),(d-1) marginal
% contract last dimension
marginalized_new{3} = tns_mult(marginalized_new{3},2,...
    legendre_weights{end},1);
% contract with second to last core
marginalized_new{2} = tns_mult(marginalized_new{2},...
    3,marginalized_new{3},1);
marginal_dm2dm1 = cell2core(tt_tensor(),...
    {marginalized_new{1};marginalized_new{2}});
marginal_dm2dm1 = full(marginal_dm2dm1,[inp(d-2,3) inp(d-1,3)]);
% for visualization, build meshgrid

[Y,X] = meshgrid(legendre_grids{d-1},legendre_grids{d-2});
figure(1);
surf(X,Y,marginal_dm2dm1); shading interp;
view(2);
xlim([-0.4 0.3]); ylim([-0.8 0]);
 
% (d-1),d marginal
% contract (d-2) dimension
marginalized_new = save_marginalized_new;
marginalized_new{1} = tns_mult(marginalized_new{1},2,...
    legendre_weights{end-2},1);
% contract with second to last core
marginalized_new{2} = tns_mult(marginalized_new{1},...
    2,marginalized_new{2},1);
marginal_dm1d = cell2core(tt_tensor(),...
    {marginalized_new{2};marginalized_new{3}});
marginal_dm1d = full(-marginal_dm1d,[inp(d-1,3) inp(d,3)]);
% for visualization, build meshgrid
[Y,X] = meshgrid(legendre_grids{d},legendre_grids{d-1});
figure(2);
surf(X,Y,-marginal_dm1d); shading interp;
view(2);
xlim([-1 0.1]); ylim([-1 1]);

%% sample
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
%%
% truncate the coefficient train
truncate = true;
if truncate
    % perform rank truncation
    max_rank = 30;
    C = round(coeff_tt, 1e-10, max_rank);
end

C = C./norm(C); % renormalize after rounding, since rounding breaks norm
% continuous tensor train samples

%%
C = C./norm(C);
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    s
    X(:,s) = get_sample(C,all_A,all_xg);
end

%% evaluate likelihood
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
    % use standard Gaussian5
    mu_samp = 0 * mean(X,2);
    % make sure in [-1,1]
    cov_samp = 0.05 * eye(d);
end

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

%%
% save the samples and exact likelihoods for normalizing flow processing
save("double_rosen3d_full_rank.mat")
