% convergence study of TT bond dimensions:
% plot dependence of relative error with 
% increasing bond dimension.

% Hongli Zhao, 12/9/2021
clear; clc; rng('default');
% GL1d
% load full rank
load("gl1d_successful.mat");
% make sure untruncated TT is normalized
coeff_tt = coeff_tt./norm(coeff_tt);
all_ranks_gl1d = 1:max(coeff_tt.r);
all_rel_err_gl1d = zeros(1,length(all_ranks_gl1d));
for i = 1:length(all_ranks_gl1d)
    i
    % truncated the TT
    max_rank = all_ranks_gl1d(i);
    C = round(coeff_tt,1e-10,max_rank);
    C = C./norm(C);
    all_rel_err_gl1d(i) = norm(coeff_tt-C)./norm(coeff_tt);
end

% GL2d
% load full rank
load("gl2d_successful.mat");
% make sure untruncated TT is normalized
coeff_tt = coeff_tt./norm(coeff_tt);
all_ranks_gl2d = 1:max(coeff_tt.r);
all_rel_err_gl2d = zeros(1,length(all_ranks_gl2d));
for i = 1:length(all_ranks_gl2d)
    i
    % truncated the TT
    max_rank = all_ranks_gl2d(i);
    C = round(coeff_tt,1e-10,max_rank);
    C = C./norm(C);
    all_rel_err_gl2d(i) = norm(coeff_tt-C)./norm(coeff_tt);
end

% Rosenbrock
% load full rank
load("./data/rosenbrock_sample.mat");
% make sure untruncated TT is normalized
coeff_tt = coeff_tt./norm(coeff_tt);
all_ranks_rosen = 1:max(coeff_tt.r);
all_rel_err_rosen = zeros(1,length(all_ranks_rosen));
for i = 1:length(all_ranks_rosen)
    i
    % truncated the TT
    max_rank = all_ranks_rosen(i);
    C = round(coeff_tt,1e-10,max_rank);
    C = C./norm(C);
    all_rel_err_rosen(i) = norm(coeff_tt-C)./norm(coeff_tt);
end

% double Rosen
% load full rank
load("./data/double_rosen_full_rank.mat");
% make sure untruncated TT is normalized
coeff_tt = coeff_tt./norm(coeff_tt);
all_ranks_drosen = 1:max(coeff_tt.r);
all_rel_err_drosen = zeros(1,length(all_ranks_drosen));
for i = 1:length(all_ranks_drosen)
    i
    % truncated the TT
    max_rank = all_ranks_drosen(i);
    C = round(coeff_tt,1e-10,max_rank);
    C = C./norm(C);
    all_rel_err_drosen(i) = norm(coeff_tt-C)./norm(coeff_tt);
end

% plot behaviors
figure(1);
plot(all_ranks_gl1d,all_rel_err_gl1d,'LineWidth',2.,'Color','blue');
title("Rel. Error Convergence for GL 1d");
grid on;
figure(2);
plot(all_ranks_gl2d,all_rel_err_gl2d,'LineWidth',2.,'Color','blue');
title("Rel. Error Convergence for GL 2d");
grid on;
figure(3);
plot(all_ranks_rosen,all_rel_err_rosen,'LineWidth',2.,'Color','blue');
title("Rel. Error Convergence for Rosenbrock");
grid on;
figure(4);
plot(all_ranks_drosen,all_rel_err_drosen,'LineWidth',2.,'Color','blue');
title("Rel. Error Convergence for Double Rosenbrock");
hold on; grid on;
save("./data/conv_study.mat");

