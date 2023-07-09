%% CTT sampling from a standard Gaussian
%   Test code: sampling from Gaussian 
clear; clc; rng("default");
p = 30;   % highest Legendre polynomial order
Ns = 10000; % number of samples
% external bond dimension for forming coeff tensor
n = p + 1;
% sample from Gaussian
d = 10;
L = 1;
legendre_ord = p+1;
% test d-dimensional Gaussian
x_sym = sym('x',[1,d]);
% scaled distribution: [-R, R]^d
R = 10;

% create a random spd matrix as covariance
covmat = randn(d,d);
covmat = covmat'*covmat;
% create random mean
mu = 10*randn(1,d)/2;

sqrt_f_sym = sqrt(mvnpdf(R*x_sym));
u_sym_temp = x_sym(:).';
f_ = matlabFunction(sqrt_f_sym, ...
    'Vars', {u_sym_temp});
% functional TT cross
[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, n, d, f_, legendre_ord, 1e-10);
C = coeff_tt;
% coefficient tensor needs to have norm 1
C = C/norm(C);

% form ultrafine grid for sampling
M = 10000; % number of ultrafine grid points
xg = linspace(-L,L,M);
dx = xg(2)-xg(1);
xg = xg+dx/2;
xg(end) = [];
% ===== added
M = length(xg);
% create design matrix
A = get_legendre(xg,p,true)';

% continuous sampling
X = zeros(d,Ns);
for s=1:Ns % this could be parfor
    X(:,s) = get_sample_copy(C,A,xg);
end
% evaluate likelihood
likes = ftt_eval2(X,C);

% compare CTT likelihood with exact likelihood
exact_likes = (R^d)*mvnpdf(R*X');
% compare analytic likelihood to CTT likelihood
ratio = log(likes./exact_likes);
% look at first 10 entries of ratio
disp(ratio(1:10))

% compare with MATLAB built-in sampler
X_matlab = mvnrnd(zeros(1,d),eye(d,d),Ns)';
% scatter to compare
compare_samples = true;
if compare_samples
    figure(1);
    subplot(1,2,1);
    scatter(R*X(1,:),R*X(2,:));
    subplot(1,2,2);
    scatter(X_matlab(1,:),X_matlab(2,:));
end

%% CTT sampling from a shifted Gaussian
clear; clc; rng("default");
p = 30;   % highest Legendre polynomial order
Ns = 2000; % number of samples
% external bond dimension for forming coeff tensor
n = p + 1;
% sample from Gaussian
d = 10;
L = 1;
legendre_ord = p+1;
% test d-dimensional Gaussian
x_sym = sym('x',[1,d]);
% scaled distribution: [-R, R]^d
R = 25;

% create a random spd matrix as covariance
covmat = randn(d,d);
covmat = covmat'*covmat;

% create random mean
mu = zeros(1,d);

sqrt_f_sym = sqrt(mvnpdf(R*x_sym,mu,covmat));
u_sym_temp = x_sym(:).';
f_ = matlabFunction(sqrt_f_sym, ...
    'Vars', {u_sym_temp});
% functional TT cross
[coeff_tt, ~] = ...
    legendre_ftt_cross_copy(L, n, d, f_, legendre_ord, 1e-10);
C = coeff_tt;
% coefficient tensor needs to have norm 1
C = C/norm(C);

% form ultrafine grid for sampling
M = 10000; % number of ultrafine grid points
xg = linspace(-L,L,M);
dx = xg(2)-xg(1);
xg = xg+dx/2;
xg(end) = [];
% ===== added
M = length(xg);
% create design matrix
A = get_legendre(xg,p,true)';

% continuous sampling
X = zeros(d,Ns);
for s=1:Ns
    X(:,s) = get_sample_copy(C,A,xg);
end
% evaluate likelihood
likes = ftt_eval2(X,C);

% compare CTT likelihood with exact likelihood
exact_likes = mvnpdf(R*X',mu,covmat);
% compare analytic likelihood to CTT likelihood
ratio = log(likes./exact_likes);
% look at first 10 entries of ratio
disp(ratio(1:10))

% compare with MATLAB built-in sampler
X_matlab = mvnrnd(mu,covmat,Ns)';

% scatter to compare
compare_samples = true;
if compare_samples
    figure(1);
    subplot(1,2,1);
    scatter(R*X(1,:),R*X(2,:));
    xlim([-20, 20]); ylim([-20, 20]);
    subplot(1,2,2);
    scatter(X_matlab(1,:),X_matlab(2,:));
    xlim([-20, 20]); ylim([-20, 20]);
end