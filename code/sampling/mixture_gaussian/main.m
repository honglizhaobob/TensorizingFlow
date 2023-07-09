%% Test sampling for Multimodal Mixture Gaussian 
% building sampler
clear; clc; rng('default');
% highest number of polynomials
p = 512;
legendre_ord = p + 1; 
% generate metadata for grid
d = 30;
inp = zeros(d,3);
for i = 1:d
    inp(i,1) = -1;
    inp(i,2) = 1;
    inp(i,3) = 2^9;   
end

% 5 modes, specify in two dimensions
mu1 = zeros(1,d);
mu2 = zeros(1,d);
mu2(d-1) = mu2(d-1)+1; % (1,1)
mu2(d) = mu2(d)+1;
mu3 = zeros(1,d);
mu3(d-1) = mu3(d-1)-1; % (-1,1)
mu3(d) = mu3(d)+1;
mu4 = zeros(1,d);
mu4(d-1) = mu4(d-1)+1; % (-1,1)
mu4(d) = mu4(d)-1;
mu5 = zeros(1,d);
mu5(d-1) = mu5(d-1)-1; % (-1,-1)
mu5(d) = mu5(d)-1;

% covariance matrices
covariance_mode = "patterned";
% 'same_random', 'diff_random'
if covariance_mode == "diff_random"
    base_mat1 = rand(d);
    base_mat1 = base_mat1'*base_mat1;

    base_mat2 = rand(d);
    base_mat2 = base_mat2'*base_mat2;

    base_mat3 = rand(d);
    base_mat3 = base_mat3'*base_mat3;

    base_mat4 = rand(d);
    base_mat4 = base_mat4'*base_mat4;

    base_mat5 = rand(d);
    base_mat5 = base_mat5'*base_mat5;
elseif covariance_mode == "same_random"
    base_mat1 = rand(d);
    base_mat1 = base_mat1'*base_mat1;
    base_mat2 = base_mat1;
    base_mat3 = base_mat1;
    base_mat4 = base_mat1;
    base_mat5 = base_mat1;
elseif covariance_mode == "patterned"
    base_mat2 = eye(d);
    base_mat2(d-1,d) = 19/20;
    base_mat2(d,d-1) = 19/20;
    
    base_mat4 = eye(d);
    base_mat4(d-1,d) = -19/20;
    base_mat4(d,d-1) = -19/20;
    
    base_mat1 = eye(d);

    base_mat3 = base_mat4;
    base_mat5 = base_mat2;
   
else
    % plain
    base_mat1 = eye(d);
    base_mat2 = eye(d);
    base_mat3 = eye(d);
    base_mat4 = eye(d);
    base_mat5 = eye(d);
end


covmat1 = 0.04*base_mat1;
covmat2 = 0.04*base_mat2;
covmat3 = 0.04*base_mat3;
covmat4 = 0.04*base_mat4;
covmat5 = 0.04*base_mat5;


% create mixture Gaussian PDF with 5 modes
sym_theta = sym('theta', [1,d]);

scaling = 2;
% use symbolic computation
raw_func = 0;

% std before scaling is sqrt(0.04)=0.2, after scaling is 0.2*2=0.4

% weight of each Gaussian (uniform weight)
gaussian_weight = 1/5;
raw_func = raw_func + gaussian_weight*mvnpdf(scaling*sym_theta,mu1,covmat1);
raw_func = raw_func + gaussian_weight*mvnpdf(scaling*sym_theta,mu2,covmat2);
raw_func = raw_func + gaussian_weight*mvnpdf(scaling*sym_theta,mu3,covmat3);
raw_func = raw_func + gaussian_weight*mvnpdf(scaling*sym_theta,mu4,covmat4);
raw_func = raw_func + gaussian_weight*mvnpdf(scaling*sym_theta,mu5,covmat5);

% sqrt of density
raw_func = sqrt(raw_func);

% convert raw function to a handle
r_func = matlabFunction(raw_func, 'Vars', {sym_theta});

% test sampling
tol = 1e-11; % set to 1e-10 for accuracy
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
    % use legendre nodes to sample
    [xg,~] = lgwt(M_xg,-1,1);
    % set grid to be ascending order
    xg = flip(xg)'; % need to be a row vector
    % nonuniform spacing
    all_spacings = abs(xg(2:end)-xg(1:end-1));
    xg = xg + ([all_spacings,0]./2); % the 0 is dummy, we don't use it
    xg(end) = [];
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
marginal_dm2dm1 = full(marginal_dm2dm1,[inp(d-3,3) inp(d-2,3)]);
% for visualization, build meshgrid

[Y,X] = meshgrid(legendre_grids{d-1},legendre_grids{d-2});

fig = figure('PaperUnits', 'inches', 'PaperSize', ...
    [8 4], 'PaperPosition', [0 0 8 4]);
surf(scaling*X,scaling*Y,scaling*marginal_dm2dm1); 
colormap turbo;
c = colorbar('westoutside');
c.Ticks=[];
shading interp;
axis equal;

ax = gca;
ax.FontSize = 20; 

xlim([-2,2]); ylim([-2,2]);
%xlabel("$x_{28}$", 'Interpreter','latex');
%ylabel("$x_{29}$", 'Interpreter','latex');
view(2);
exportgraphics(fig,'./img/mixture_dm2dm1.jpg','Resolution',500);
%%
 
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
marginal_dm1d = full(marginal_dm1d,[inp(d-1,3) inp(d,3)]);
% for visualization, build meshgrid
[Y,X] = meshgrid(legendre_grids{d},legendre_grids{d-1});
figure(2);

fig = figure('PaperUnits', 'inches', 'PaperSize', ...
    [8 4], 'PaperPosition', [0 0 8 4]);
surf(scaling*X,scaling*Y,marginal_dm1d); 
shading interp;
colormap turbo;
c = colorbar('westoutside');
c.Ticks=[];
shading interp;
axis equal;

ax = gca;
ax.FontSize = 20; 

xlim([-2,2]); ylim([-2,2]);
%xlabel("$x_{29}$", 'Interpreter','latex');
%ylabel("$x_{30}$", 'Interpreter','latex');
view(2);
exportgraphics(fig,'./img/mixture_dm1d.jpg','Resolution',500);
%%
% truncate the coefficient train
truncate = true;
if truncate
    % perform rank truncation
    max_rank = 2;
    C = round(coeff_tt, 1e-10, max_rank);
end

C = C./norm(C); % renormalize after rounding, since rounding breaks norm
% continuous tensor train samples
%%
C = C./norm(C);
X = zeros(d,Ns);
for s=1:Ns
    X(:,s) = get_sample(C,all_A,all_xg);
end
save_X = X;
%% After sampling, add smearing in each dimension
X = save_X;
for s = 1:Ns
    for i = 1:d
        xg_i = all_xg{i};
        legendre_idx = find(xg_i <= X(i,s), 1, 'last');
        if isempty(legendre_idx)
           legendre_idx = 1; 
        end
        spacings = xg_i(2:end)-xg_i(1:end-1);
        % to prevent error, perturb the last point the same way
        % as (N-1)th point
        if legendre_idx >= length(spacings)
            legendre_idx = legendre_idx - 1;
        end
        dx_i = spacings(legendre_idx);
        X(i,s) = X(i,s)+2*((dx_i*randn)-(dx_i/2));
    end
end

%% Plot scatter plot after sampling
% visualize samples

% rescale all samples in each dimension
new_X = X;
for dim = 1:d
    new_X(dim,:) = scaling.*new_X(dim,:);
end

% plot result
figure(1);
scatter(new_X(end-2,:), new_X(end-1,:), 5, 'blue');
grid on;
figure(2);
scatter(new_X(end-1,:), new_X(end,:), 5, 'blue');

grid on;
%%
% continuous tensor train samples
X = zeros(d,Ns);
for s=1:Ns
    X(:,s) = get_sample(C,all_A,all_xg);
end

% visualize samples
figure(1);
scatter(X(end-2,:), X(end-1,:), 5, 'blue');
grid on;

figure(3);
scatter(X(end-1,:), X(end,:), 5, 'blue');
grid on;

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
% save the samples and exact likelihoods for normalizing flow processing
%%
%save("./data/mixture_gaussian_full_rank.mat");
save("./data/mixture_gaussian_patterned_truncated.mat");

%% Sanity check 
% Make sure data is loaded
sample_point = ones(1,d);
% MATLAB mvnpdf
matlab_value = log(mvnpdf(sample_point, mu1, covmat1) +...
    mvnpdf(sample_point, mu2, covmat2) ...
    + mvnpdf(sample_point, mu3, covmat3) + ...
    + mvnpdf(sample_point, mu4, covmat4) + ...
    mvnpdf(sample_point, mu5, covmat5));
% compute mvnpdf manually
manual_value = log((1/(sqrt((2*pi)^d)*abs(det(covmat1))))*...
    exp(-0.5*(sample_point-mu1)*inv(covmat1)*(sample_point-mu1)') + ...
    (1/(sqrt((2*pi)^d)*det(covmat2)))*...
    exp(-0.5*(sample_point-mu2)*inv(covmat2)*(sample_point-mu2)') + ...
    (1/(sqrt((2*pi)^d)*det(covmat3)))*...
    exp(-0.5*(sample_point-mu3)*inv(covmat3)*(sample_point-mu3)') + ...
    (1/(sqrt((2*pi)^d)*det(covmat4)))*...
    exp(-0.5*(sample_point-mu4)*inv(covmat4)*(sample_point-mu4)') + ...
    (1/(sqrt((2*pi)^d)*det(covmat5)))*...
    exp(-0.5*(sample_point-mu5)*inv(covmat5)*(sample_point-mu5)'));
% compute manually, use LSE trick
c1 = log(1/(sqrt((2*pi)^d)*abs(det(covmat1))));
c2 = log(1/(sqrt((2*pi)^d)*abs(det(covmat2))));
c3 = log(1/(sqrt((2*pi)^d)*abs(det(covmat3))));
c4 = log(1/(sqrt((2*pi)^d)*abs(det(covmat4))));
c5 = log(1/(sqrt((2*pi)^d)*abs(det(covmat5))));
x1 = -0.5*(sample_point-mu1)*inv(covmat1)*(sample_point-mu1)';
x2 = -0.5*(sample_point-mu2)*inv(covmat2)*(sample_point-mu2)';
x3 = -0.5*(sample_point-mu3)*inv(covmat3)*(sample_point-mu3)';
x4 = -0.5*(sample_point-mu4)*inv(covmat4)*(sample_point-mu4)';
x5 = -0.5*(sample_point-mu5)*inv(covmat5)*(sample_point-mu5)';
x_max = max([c1+x1, c2+x2, c3+x3, c4+x4, c5+x5]);
lse_manual = x_max + log(exp(x1-x_max) + exp(x2-x_max) +...
    exp(x3-x_max) + exp(x4-x_max) +...
    exp(x5-x_max));




