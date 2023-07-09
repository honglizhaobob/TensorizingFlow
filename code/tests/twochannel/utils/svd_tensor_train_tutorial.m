%% Check SVD with Different Kernels
% ===============
%  Test 1: d = 3
% ===============
% let p(x1,x2,x3) = K1(x1,x2)*K2(x2,x3) where K1 and K2 are different 
clear; clc; rng('default');
k = 200; % number of grid points
L = -2; 
grid = linspace(-L,L,k);
dx = grid(2)-grid(1); x_start = grid(1); x_end = grid(end);
% create exact function

% f = sin(x1+x2) * cos(x2*exp(x3))
exact_tensor = zeros(k,k,k);
for i1 = 1:k
    for i2 = 1:k
        for i3 = 1:k
            x1 = grid(i1); 
            x2 = grid(i2); 
            x3 = grid(i3);
            exact_tensor(i3,i2,i1) = sin(x1+2*x2) * cos(x2*exp(x3));
        end
    end
end

% create SVD tensor train

% kernels
K1 = sin(grid'+2*grid);
K2 = cos(grid'.*exp(grid));
% do SVD on both kernels
[u1,s1,v1] = svd(K1);
[u2,s2,v2] = svd(K2);
a1 = u1*s1; b1 = v1;
a2 = u2*s2; b2 = v2;
% truncate
idx1 = find(diag(s1)./max(diag(s1))>1e-16);
idx2 = find(diag(s2)./max(diag(s2))>1e-16);
r1 = length(idx1); r2 = length(idx2);
a1_r = a1(:,idx1); b1_r = b1(:,idx1); 
a2_r = a2(:,idx2); b2_r = b2(:,idx2);
disp(strcat("> Error from SVD: K1 = ", num2str(norm(a1_r*b1_r'-K1)),...
    " , K2 = ", num2str(norm(a2_r*b2_r'-K2))))
% build cores
left_core = reshape(b2_r, [1 size(b2_r)]);
right_core = reshape(a1_r', [size(a1_r') 1]);
mid_core = zeros(r2,k,r1);
for i = 1:r1
    for j = 1:r2
        mid_core(j,:,i) = a2_r(:,j).*b1_r(:,i);
    end
end
% build TT
svd_tensor = cell2core(tt_tensor, {left_core; mid_core; right_core});
disp(strcat("> Error (SVD TT) - (Exact Tensor) = ", ...
    num2str(norm(full(svd_tensor)-exact_tensor(:)))));
%%
% ===============
%  Test 2: d = 4
% ===============
clear; clc; rng('default');
% p(x1,x2,x3,x4) = K1(x1,x2) * K2(x2,x3) * K2(x3,x4) where K1, K2 are
% different.
k = 50;
L = -2; 
grid = linspace(-L,L,k); dx = grid(2)-grid(1); 
x_start = grid(1); x_end = grid(end);
% create exact tensor
% f = sin(x1 + 2*x2) * exp(x2 - 4*cos(x3)) * exp(x3 - 4*cos(x4))
exact_tensor = zeros(k, k, k, k);
for i1 = 1:k
    for i2 = 1:k
        for i3 = 1:k
            for i4 = 1:k
                x1 = grid(i1); x2 = grid(i2);
                x3 = grid(i3); x4 = grid(i4);
                exact_tensor(i1,i2,i3,i4) = ...
                    sin(x1+2*x2)*exp(x2-4*cos(x3))*exp(x3-4*cos(x4));
            end
        end
    end
end

% build TT using SVD
K1 = sin(grid' + 2 * grid);
K2 = exp(grid' - 4 * cos(grid));
% SVD
[u1,s1,v1] = svd(K1);
[u2,s2,v2] = svd(K2);
a1 = u1*s1; b1 = v1;
a2 = u2*s2; b2 = v2;
% truncate
idx1 = find(diag(s1)./max(diag(s1)) > 1e-15);
idx2 = find(diag(s2)./max(diag(s2)) > 1e-15);
r1 = length(idx1); r2 = length(idx2);
a1_r = a1(:,idx1); b1_r = b1(:,idx1);
a2_r = a2(:,idx2); b2_r = b2(:,idx2);
disp(strcat("> Error from Kernel SVD, K1 = ", ...
    num2str(norm(a1_r*b1_r'-K1)), ", K2 = ", ...
    num2str(norm(a2_r*b2_r'-K2))))
% build cores
mid_core1 = zeros(r1,k,r2);
mid_core2 = zeros(r2,k,r2);
for i = 1:r1
    for j = 1:r2
        mid_core1(i,:,j) = b1_r(:,i).*a2_r(:,j);
    end
end

for i = 1:r2
    for j = 1:r2
        mid_core2(i,:,j) = b2_r(:,i).*a2_r(:,j);
    end
end

left_core = reshape(a1_r, [1 size(a1_r)]);
right_core = reshape(b2_r', [size(b2_r') 1]);
% build TT
svd_tt = cell2core(tt_tensor, {left_core; ...
    mid_core1; mid_core2; right_core});
disp(strcat("> Error (SVD TT) and (Exact Tensor) = ", ...
    num2str(norm(full(svd_tt)-exact_tensor(:)))))
%% Check low-dimensional exact tensor with SVD TT
clear; clc; rng("default");
beta = 4;
N = 3; 
dt = 1.5e-3;
t_start = 0; t_end = t_start + N*dt;
% spatial domain
k = 10; 
L = 2; X_arr = linspace(-L, L, k);

U = sym('u', [N, 2], 'real');
path_action = energy_potential(U(:), dt, beta);
u_sym_temp = U(:);
u_sym_temp = u_sym_temp';
normal_c = 2.1893e+05^2;
energy_func = matlabFunction((1/normal_c)*path_action, 'Vars', {u_sym_temp});

% first build exact tensor
energy_tensor_exact = zeros(repmat(k, [1,2*N]));

for i1 = 1:k
    u1_1 = X_arr(i1);
    for i2 = 1:k
        u1_2 = X_arr(i2);
        for i3 = 1:k
            u2_1 = X_arr(i3);
            for i4 = 1:k
                u2_2 = X_arr(i4);
                for i5 = 1:k
                    u3_1 = X_arr(i5);
                    for i6 = 1:k
                        u3_2 = X_arr(i6);
                        path = [u1_1, u1_2; ...
                                u2_1, u2_2; ...
                                u3_1, u3_2];
                        energy_tensor_exact(i1,i2,i3,i4,i5,i6) = ...
                            energy_func(path(:).');
                    end
                end
            end
        end
    end
end

% compare exact energy tensor with amen_cross
X_tt = tt_tensor(X_arr);
X = tt_meshgrid_vert(X_tt, N*2);
energy_amen_tt = amen_cross_s(X, energy_func, 1e-100, 'nswp', 50);
energy_tensor_approx = full(energy_amen_tt, repmat(k, [1,2*N]));
norm(energy_tensor_approx(:)-energy_tensor_exact(:))
max(energy_tensor_exact(:))
%% Test amen_cross is inflenced by large number
clear; clc; rng("default");

% ising model grid
%grid = [-1, 1]; 

% general function grid
grid = linspace(-2,2,20);

k = length(grid);
dim = 4; 
X = sym('x', [1,dim], 'real');
tmp = X(:).';
large_number = 1;
func = large_number * exp(-( (sin(X(2))-(cos(X(1).^2)-2)) + ...
    (sin(X(3))-(cos(X(2).^2)-2)) + ...
   (sin(X(4))-(cos(X(3).^2)-2)) ));

% 2d ising model
%T = 1;
%func = exp(-T*X(1)*X(2));
% 3d ising model
%func = exp(-T*X(1)*X(2)) .* exp(-T*X(2)*X(3));

% 4d ising model
%func = exp(-T*X(1)*X(2)) .* exp(-T*X(2)*X(3)) .* exp(-T*X(3)*X(4));

func = matlabFunction(func, 'Vars', {tmp});
% exact tensor
% 2d ising model
%exact_tensor = zeros(k,k);
%for i1 = 1:k
%    x1 = grid(i1);
%    for i2 = 1:k
%        x2 = grid(i2);
%        exact_tensor(i1, i2) = func([x1 x2]);
%    end
%end

% 3d ising model
%for i1 = 1:k
%    x1 = grid(i1);
%    for i2 = 1:k
%        x2 = grid(i2);
%        for i3 = 1:k
%            x3 = grid(i3);
%            exact_tensor(i1,i2,i3) = func([x1, x2, x3]);
%        end
%    end
%end
        
% 4d ising model
for i1 = 1:k
    x1 = grid(i1);
    for i2 = 1:k
        x2 = grid(i2);
        for i3 = 1:k
            x3 = grid(i3);
            for i4 = 1:k
                x4 = grid(i4);
                exact_tensor(i1,i2,i3,i4) = func([x1, x2, x3, x4]);
            end
        end
    end
end


% look at each matrix for each i3
see_matrix = false;
if see_matrix
    for i = 1:k
        for j = 1:k
            figure(1); hold on;
            xlim([0,k]); ylim([0,k]);
            imagesc(squeeze(exact_tensor(:,:,i,j)));
        end
    end
end

% amen_cross
X_tt = tt_tensor(grid);
X = tt_meshgrid_vert(X_tt, 4);
test_tt = amen_cross(X, func, 1e-10, 'nswp', 30);
approx_tensor = full(test_tt, [k k k k]);
% look at each matrix for each i3
if see_matrix
    for i = 1:k
        for j = 1:k
            figure(2); hold on;
            xlim([0,k]); ylim([0,k]);
            imagesc(squeeze(approx_tensor(:,:,i,j)));
        end
    end
end
% evaluate difference in norm
disp("=== fro norm approx error")
norm(approx_tensor(:)-exact_tensor(:))
norm(full(test_tt)-exact_tensor(:))

% build the function TT using SVD
disp("===== building tensor train using SVD: ")
% general kernel
K = sin(grid).'-(cos(grid.^2)-2);
K = exp(-K);

% for ising model
%K = grid .* grid.';
%K = exp(-T*K);
[u,s,v] = svd(K);
A = u*s; B = v;
idx = find(diag(s)/max(diag(s))>1e-10);
r = length(idx);
core = zeros(r,k,r);
a_r = A(:,idx); b_r = B(:,idx);
norm(a_r*b_r'-K)

for i=1:r
    for j=1:r
        core(i,:,j) = a_r(:,i).*b_r(:,j);
    end
end

% form SVD core
% 2d ising model
%test_svd_tt = cell2core(tt_tensor, ...
%    {reshape(a_r, [1,k,r]); ...
%    reshape(b_r.', [r,k,1])});

% 3d ising model
%test_svd_tt = cell2core(tt_tensor, ...
%    {reshape(b_r, [1,k,r]); ...
%    reshape(core, [r,k,r]); ...
%    reshape(a_r', [r,k,1])});

% 4d ising model
test_svd_tt = cell2core(tt_tensor, ...
    {reshape(b_r, [1,k,r]); ...
    reshape(core, [r,k,r]); ...
    reshape(core, [r,k,r]);
    reshape(a_r', [r,k,1])});


approx_tensor2 = full(test_svd_tt, [k k k k]);
exact_tt = tt_tensor(exact_tensor);
% look at each matrix for each i3
%for i = 1:k
%    figure(2); hold on;
%    xlim([0,k]); ylim([0,k]);
%    imagesc(approx_tensor2(:,:,i));
%end
norm(approx_tensor2(:) - exact_tensor(:))
norm(test_svd_tt - exact_tt)
%%
% example in R^2 (two spatial points: X1, X2)
clear; clc; rng('default')
k = 10;
grid = linspace(-2,2,k);
[X,Y] = meshgrid(grid);
% flatten by column
x2d = [X(:) Y(:)];
% parameter
dt = 1;
% create function 
X = sym('x', [2,2], 'real');
func = exp(- (1/dt).* norm(X(2,:)-X(1,:))^2 );
tmp = X(:).';
func = matlabFunction(func, 'Vars', {tmp});
% build exact tensor 
exact_tensor = zeros(k,k,k,k);
for i1 = 1:k
    x1_1 = grid(i1);
    for i2 = 1:k
        x2_1 = grid(i2);
        for i3 = 1:k
            x1_2 = grid(i3);
            for i4 = 1:k
                x2_2 = grid(i4);
                path = [x1_1, x1_2; ...
                        x2_1, x2_2];
                exact_tensor(i1,i4,i2,i3) = func(path(:).');
            end
        end
    end
end

% test amen cross
X_tt = tt_tensor(grid); X = tt_meshgrid_vert(X_tt, 4);
test_tt = amen_cross_s(X, func, 1e-10, 'nswp', 30);
disp(strcat("==== norm error using AMEN_CROSS : ", ...
    num2str(norm(full(test_tt)-exact_tensor(:)))))


% test SVD
% kernel

% rows are X1, cols are X2
K = exp(-(1/dt) * ( (x2d(:,1).'-x2d(:,1)).^2 + ...
    (x2d(:,2).'-x2d(:,2)).^2 ));

[u,s,v] = svd(K);
% decouple X1,X2
a = u*s; b = v;
idx = find(diag(s)./max(diag(s)) > 1e-10);
r1 = length(idx)
a_r1 = a(:,idx); b_r1 = b(:,idx);
disp(strcat("==== norm error of truncated SVD (X1, X2): ", ...
    num2str(norm(a_r1*b_r1'-K))))
%core1 = reshape(b_r1, [1 k^2 r1]);
%core2 = reshape(a_r1', [r1 k^2 1]);
core1 = reshape(a_r1, [1 k^2 r1]);
core2 = reshape(b_r1', [r1 k^2 1]);
% decouple x1_1, x1_2
core1 = reshape(core1, [1*k k*r1]);
[u,s,v] = svd(core1);
a_x1 = u*s; b_x1 = v;
idx = find(diag(s)./max(diag(s))>1e-10);
r2 = length(idx)
a_r_x1 = a_x1(:,idx); b_r_x1 = b_x1(:,idx);
disp(strcat("==== norm error of truncated SVD (X1_1, X1_2): ", ...
    num2str(norm(a_r_x1*b_r_x1'-core1))))

svd_core_left = reshape(a_r_x1, [1 k r2]);
svd_core_mid1 = reshape(b_r_x1', [r2 k r1]);

% decouple x2_1, x2_2
core2 = reshape(core2, [r1*k k*1]);
[u,s,v] = svd(core2);
a_x2 = u*s; b_x2 = v;
idx = find(diag(s)./max(diag(s))>1e-10);
r3 = length(idx)
a_r_x2 = a_x2(:,idx); b_r_x2 = b_x2(:,idx);
disp(strcat("==== norm error of truncated SVD (X2_1, X2_2): ", ...
    num2str(norm(a_r_x2*b_r_x2'-core2))))
svd_core_mid2 = reshape(a_r_x2, [r1 k r3]);
svd_core_right = reshape(b_r_x2', [r3 k 1]);

% build SVD TT
svd_tt = cell2core(tt_tensor, ...
    {svd_core_left; ...
     svd_core_mid1; ...
     svd_core_mid2; ...
     svd_core_right});
 
disp(strcat("==== norm error using SVD TT : ", ...
    num2str(norm(full(svd_tt) - exact_tensor(:)))))
norm(tt_tensor(exact_tensor)-svd_tt)
norm(test_tt-svd_tt)
%disp([full(svd_tt), exact_tensor(:)]);
%% example in R^2, 3 spatial points X1 X2 X3
clear; clc; rng('default');
k = 10;
L = 1.8;
grid = linspace(-L,L,k);
[X,Y] = meshgrid(grid);
% flatten by column
x2d = [X(:) Y(:)];
% parameter 
dt = 1; % dt = 1e-2 shows advantage of using SVD
% create function
X = sym('x', [3,2], 'real');

% ==========
% Test 1 & 2
% ==========

% uncomment to run

% experiment 1
%func_sym = exp(-( norm( (1/dt)*(X(3,:)-X(2,:)) ).^2 +...
%    norm((1/dt)*(X(2,:)-X(1,:))).^2 ) );

% experiment 2: add nonlinear term
drift = @(x) [-x(:,2) x(:,1)];
func_sym = exp(-( norm( (1/dt)*(X(3,:)-X(2,:)) - drift(X(2,:)) ).^2 +...
    norm((1/dt)*(X(2,:)-X(1,:)) - drift(X(1,:)) ).^2 ) );
tmp = X(:).';
func = matlabFunction(func_sym, 'Vars', {tmp});
% build exact tensor 
exact_tensor = zeros(k,k,k,k,k,k);
for i1 = 1:k
    x1_1 = grid(i1);
    for i2 = 1:k
        x2_1 = grid(i2);
        for i3 = 1:k
            x3_1 = grid(i3);
            for i4 = 1:k
                x1_2 = grid(i4);
                for i5 = 1:k
                    x2_2 = grid(i5);
                    for i6 = 1:k
                        x3_2 = grid(i6);
                        path = [x1_1, x1_2; ...
                                x2_1, x2_2; ...
                                x3_1, x3_2];
                        %exact_tensor(i1,i2,i3,i4,i5,i6) = func(path(:).');
                        exact_tensor(i6,i3,i5,i2,i4,i1) = func(path(:).');
                    end
                end
            end
        end
    end
end
              
% test AMEN_CROSS
X_tt = tt_tensor(grid); X = tt_meshgrid_vert(X_tt, 6);
test_tt = amen_cross_s(X, func, 1e-10, 'nswp', 30);
disp(strcat("==== norm error using AMEN_CROSS (unordered) : ", ...
    num2str(norm(full(test_tt)-exact_tensor(:)))))
% test SVD
% rows are X1, cols are X2: K(i,j) = exp(-(norm(x2d(j,:)-x2d(i,:))^2)/dt)
% experiment 1
%K = exp(-(1/dt^2) * ( ( ( x2d(:,1).'-x2d(:,1) ) ).^2 + ...
%    ( x2d(:,2).'-x2d(:,2) ).^2 ));
% experiment 2
all_drift = drift(x2d);
% ==========
%   Kernel
% ==========
K = exp(-(1/dt^2) * ( ( x2d(:,1).' - (x2d(:,1) + dt*all_drift(:,1)) ).^2 + ...
    ( x2d(:,2).' - (x2d(:,2) + dt*all_drift(:,2)) ).^2 ) );
% ==========


% get common core
[u,s,v] = svd(K);
a = u*s; b = v;
idx = find(diag(s)./max(diag(s))>1e-10); 
r = length(idx);
a_r = a(:,idx); b_r = b(:,idx); disp(norm(a_r*b_r'-K))
common_core = zeros(r,k^2,r);
for i = 1:r
    for j = 1:r
        common_core(i,:,j) = a_r(:,i).*b_r(:,j);
    end
end

% boundary cores
core_left = reshape(b_r, [1 k^2 r]);
core_right = reshape(a_r', [r k^2 1]);

% decouple (X1_1, X1_2)
core_left = reshape(core_left, [1*k, k*r]);
[u,s,v] = svd(core_left);
a = u*s; b = v;
idx = find(diag(s)./max(diag(s))>1e-10); 
r0 = length(idx);
a_r0 = a(:,idx); b_r0 = b(:,idx); disp(norm(a_r0*b_r0'-core_left))
svd_left_core1 = reshape(a_r0, [1 k r0]);
svd_left_core2 = reshape(b_r0', [r0 k r]);
% decouple (X2_1, X2_2) (common core)
common_core = reshape(common_core, [r*k k*r]);
[u,s,v] = svd(common_core);
a = u*s; b = v;
idx = find(diag(s)./max(diag(s))>1e-10); 
r1 = length(idx);
a_r1 = a(:,idx); b_r1 = b(:,idx); disp(norm(a_r1*b_r1'-common_core))
svd_mid_core1 = reshape(a_r1, [r k r1]);
svd_mid_core2 = reshape(b_r1', [r1 k r]);

% decouple (X3_1, X3_2)
core_right = reshape(core_right, [r*k k*1]);
[u,s,v] = svd(core_right);
a = u*s; b = v;
idx = find(diag(s)./max(diag(s))>1e-10);
r2 = length(idx);
a_r2 = a(:,idx); b_r2 = b(:,idx); disp(norm(a_r2*b_r2'-core_right))
svd_right_core1 = reshape(a_r2, [r k r2]);
svd_right_core2 = reshape(b_r2', [r2 k 1]);

% build SVD TT
svd_tt = cell2core(tt_tensor, ...
    {svd_left_core1; svd_left_core2; ...
     svd_mid_core1;  svd_mid_core2;  ...
     svd_right_core1; svd_right_core2});
% check norm 
disp(strcat("==== norm of SVD TT = ", num2str(norm(full(svd_tt)))))
disp(strcat("==== norm of exact tensor = ", num2str(norm(exact_tensor(:)))))
disp(strcat("==== norm error using SVD TT : ", ...
    num2str(norm(full(svd_tt) - exact_tensor(:)))))
%v1 = full(svd_tt); v2 = exact_tensor(:);
%disp([v1 v2])

% the ordering of svd tt's variables is:
% (x3_2, x3_1, x2_2, x2_1, x1_2, x1_1)
% rebuild FUNC
tmp = reshape(tmp, [3 2]).';
tmp = flip(tmp(:)).';
func = matlabFunction(func_sym, 'Vars', {tmp});
% now requires many more sweeps
test_tt = amen_cross_s(X, func, 1e-10, 'nswp', 100);
disp(strcat("==== norm error using AMEN_CROSS (reordered) and EXACT : ", ...
    num2str(norm(full(test_tt)-exact_tensor(:)))))
disp(strcat("==== norm error btwn AMEN_CROSS (reordered) and SVD : ", ...
    num2str(norm(full(test_tt)-full(svd_tt)))))
disp(strcat("==== norm error btwn EXACT and SVD : ", ...
    num2str(norm(full(svd_tt)-exact_tensor(:)))))
%% Run previous code section
% ==========
%   Test 3
% ==========
% use the same cores, test a higher-dimensional example
% can only test error btwn SVD TT and AMEN_CROSS TT

% if dt too small, may not work as most entries are 0
assert(dt==1, "Please re-run the above tests to create a denser kernel ");
drift;
% set number of dimensions (effective dimension is double this)
d = 2;
assert(d>=2, "Must have at least 2 points in R^2. ");
X = sym('x', [2*d,2], 'real');
% create exact function for AMEN_CROSS
func_sym = exp(  -sum(sum(( (1/dt)*(X(2:end,:)-X(1:end-1,:)) ...
    - drift(X(1:end-1,:)) ).^2))  );
func_sym2 = exp(-( norm( (1/dt)*(X(3,:)-X(2,:)) - drift(X(2,:)) ).^2 +...
    norm((1/dt)*(X(2,:)-X(1,:)) - drift(X(1,:)) ).^2 ...
    + norm((1/dt)*(X(4,:)-X(3,:)) - drift(X(3,:)) ).^2) ); % for d=2
x1_1 = rand(); x1_2 = rand(); x2_1 = rand(); x2_2 = rand();
x3_1 = rand(); x3_2 = rand(); x4_1 = rand(); x4_2 = rand();
assert(eval(func_sym - func_sym2) == 0);
% ==========
% change for higher dimensions
% ==========
d = 5;
X = sym('x', [d 2], 'real');
% create exact function for AMEN_CROSS
func_sym = exp(  -sum(sum(( (1/dt)*(X(2:end,:)-X(1:end-1,:)) ...
    - drift(X(1:end-1,:)) ).^2))  );
tmp = X(:).';
func = matlabFunction(func_sym, 'Vars', {tmp});
% create AMEN_CROSS tensor
X_tt; 
X = tt_meshgrid_vert(X_tt, 2*d);
test_tt = amen_cross_s(X, func, 1e-10, 'nswp', 50);
disp(strcat("===== norm of AMEN TT (unordered) = ",...
    num2str(norm(test_tt))));
% create reordered AMEN_CROSS tensor
% according to (Xd_2, Xd_1, ..., X2_2, X2_1, X1_2, X1_1)
tmp = reshape(tmp, [d 2]).';
tmp = flip(tmp(:)).';
func = matlabFunction(func_sym, 'Vars', {tmp});
% treat as exact TT for norm comparison
test_tt_amen_exact = amen_cross_s(X, func, 1e-10, 'nswp', 50);
disp(strcat("===== norm of AMEN TT (reordered) = ",...
    num2str(norm(test_tt_amen_exact))));

% create SVD TT (kernel does not change from Test 1&2, need to
% repeat common_core

% create cell containing all cores, to pass into
% TT_TENSOR constructor
all_cores_cell = cell(2*d,1);
all_cores_cell{1} = svd_left_core1;
all_cores_cell{2} = svd_left_core2;
all_cores_cell{end-1} = svd_right_core1;
all_cores_cell{end} = svd_right_core2;
for i = 2:d-1
    % put common core
    all_cores_cell{2*i-1} = svd_mid_core1;
    all_cores_cell{2*i} = svd_mid_core2;
end
% create SVD TT
% variable ordering (Xd_2, Xd_1, ..., X2_2, X2_1, X1_2, X1_1)
svd_tt = cell2core(tt_tensor, all_cores_cell);
% compare norm error
disp("==========")
disp("  Report  ")
disp("==========")
disp(strcat("> norm of AMEN_CROSS TT : ",...
    num2str(norm(test_tt_amen_exact))))
disp(strcat("> norm of SVD TT : ", ...
    num2str(norm(svd_tt))))
disp(strcat("> (d = ", num2str(d), "), ", ...
    " norm error btwn AMEN_CROSS (reordered) and SVD : ", ...
    num2str(norm(test_tt_amen_exact-svd_tt))))