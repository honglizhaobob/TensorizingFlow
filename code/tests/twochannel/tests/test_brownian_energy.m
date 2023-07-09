% Main script to test that MATLAB and Python
% implementations of the Brownian bridge energy
% match for samples generated from TT.
clear; clc; rng('default');

% load generated data
load("./data/pathsampling_fullrank_small.mat");
all_energy = zeros(Ns,1);
for i = 1:Ns
    i
    % get path sample
    sample_path = R*X(:,i)';
    % compute energy
    all_energy(i) = energy_potential_brownian_bridge(sample_path, ...
        dt, beta);
end
% save energy values to be verified with Python script
save("./test_brownian_energy.mat", "all_energy");

%% Test transition path energy
clear; clc; rng('default');
% load generated data
load("./data/pathsampling_fullrank_small.mat");
all_energy = zeros(Ns,1);
for i = 1:Ns
    i
    % get path sample
    sample_path = R*X(:,i)';
    % compute energy
    all_energy(i) = energy_potential(sample_path, ...
        dt, beta);
end
% save energy values to be verified with Python script
save("./test_tpt_energy.mat", "all_energy");

