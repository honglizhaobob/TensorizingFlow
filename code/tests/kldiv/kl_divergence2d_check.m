% Hongli Zhao, honglizhaobob@uchicago.edu

% Main script to do a few sanity checks

%% Check whether KL divergences are comparable comparing
% 10000 samples from one mode, and 10000 samples from both
% modes of a bimodal Gaussian. Target distribution would be
% double mode Gaussian.

clear; clc; rng('default');

mu = [0 3;0 -3];
sigma = cat(3,[1 0;0 1],[1 0;0 1]);
p = ones(1,2)/2;
gm = gmdistribution(mu,sigma,p);
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);
figure(1);
fcontour(gmPDF,[-10 10]);
title('Contour lines of pdf');

% number of samples
N = 200000;
X_double = random(gm,N);
figure(2);


% sample from a single Gaussian
X_single = mvnrnd([0 3], [1 0;0 1], N);
scatter(X_double(:,1),X_double(:,2),3,'red'); hold on;
scatter(X_single(:,1),X_single(:,2),3,'blue');

% compute KL divergence between all samples at a single mode
% and two modes target
prior_pdf = mvnpdf(X_single,[0 3],[1 0;0 1]);
posterior_pdf = gm.pdf(X_single);
mean(log(prior_pdf) - log(posterior_pdf))





