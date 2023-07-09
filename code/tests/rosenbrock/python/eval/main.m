% main driver code to check evaluation 
% is working for all examples.

clear; clc; rng('default');

% load data for rosenbrock
load("double_rosen_data.mat");
% evaluate points
p_matlab = zeros(1,size(X,2)); 
log_p_matlab = zeros(1,size(X,2)); 
X = X';
for i = 1:size(X,1)
    p_matlab(i) = double_rosen(X(i,:));
    log_p_matlab(i) = log_double_rosen(X(i,:));
end
% compare with Python
load("double_rosen_python_result.mat")
disp(strcat(['>>> (Double Rosen) norm error between MATLAB and Python = ', ...
    num2str(norm(log_p_matlab - log_p_python))]))
%%

% load data for rosenbrock
clear; rng('default');
load("rosen_data.mat")
% evaluate points
p_matlab = zeros(1,size(X,2)); 
log_p_matlab = zeros(1,size(X,2)); 
X = X';
for i = 1:size(X,1)
    p_matlab(i) = rosen(X(i,:));
    log_p_matlab(i) = log_rosen(X(i,:));
end
% compare with Python
load("rosen_python_result.mat")
disp(strcat(['>>> (Rosen) norm error between MATLAB and Python = ', ...
    num2str(norm(log_p_matlab - log_p_python))]))