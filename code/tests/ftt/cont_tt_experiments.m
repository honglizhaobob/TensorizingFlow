% Numerical experiments for continuous TT decomposition
%% Test 1 - Gauss-Legendre Integration
clear; clc; rng('default');
% generate Gauss-Legendre collocation points on [-2,2]
L = 2;
% number of grid points
k = 20;
[X,W] = lgwt(k,-L,L);

% function to integrate
f = @(x) x.*sin(x);
F = @(x) -x.*cos(x)+sin(x); % antiderivative
% error convergence of integral over [-L,L]
int_exact = F(L)-F(-L);
int_approx = W'*f(X);
disp(strcat(["> Numerical Integration Abs. Error: ", ...
    num2str(abs(int_exact - int_approx))]));

% plot convergence error varying k
k = 1:2000;
all_err = zeros([1,length(k)]);
for i = k
    i
    % get GL points and weights
    [X,W] = lgwt(i,-L,L);
    all_err(i) = abs(int_exact - W'*f(X));
end
% plot error
figure(1); grid on; 
plot(k,log10(all_err),'Color','red','LineWidth',2);
title("Error Convergence of Gauss-Legendre Integral vs. Discretization Level");

%% Test 2: Expand a univariate function in basis
clear; clc; rng("default");
% grid [-L,L]
L = 3;
% chebyshev polynomial is rescaled by L
cheby_ord = 50; % order of Chebyshev polynomials
%Form basis for the committor function q
basisq = cell(1,cheby_ord);
for i=0:cheby_ord-1
   basisq{i+1}   = @(x) chebyshevT(i,x/L);
   visualize = false;
   if visualize
       plot((-L:0.01:L)',basisq{i+1}((-L:0.01:L)'))
       hold on;
   end
end

% univariate function to build expansion for
%f = @(x) exp(-0.25 * (x-2*sin(x.^2)).^2);
f = @(x) sin(x.^2)+exp(-x);
k = 100; % number of spatial discretization
[X,W] = lgwt(k,-L,L);
figure(1)
plot(X,f(X))

% build data, of size (k x cheby_ord), rows enumerate
% collocation points, cols enumerate basis functions
Phi = zeros([k,cheby_ord]);
for i = 1:cheby_ord
    % evaluate phi_j at collocation points
    Phi(:,i) = basisq{i}(X);
end

% evaluate original function f
f_exact = f(X);

% find Moore-Penrose left psedoinverse
coeffs = (Phi*Phi')\f_exact;
coeffs = Phi'*coeffs;

norm(Phi*coeffs-f_exact)
figure(2)
plot(X,Phi*coeffs)
%% Test 3: Expand a univariate function with Legendre basis
% Gauss-Legendre basis
clear; clc; rng('default');
L = 3;
% number of grid points
k = 100;
[X,W] = lgwt(k,-L,L);
% univariate function to build expansion for
%f = @(x) exp(-0.25 * (x-2*sin(x.^2)).^2);
f = @(x) sin(x.^2)+exp(-x);
%f = @(x) sin(x.^2);
%f = @(x) x.^2;
%plot(X,f(X));

% Legendre basis on a general domain: 
% https://math.stackexchange.com/questions/1924557...
%/legendre-polynomial-on-interval-a-b-other-than-1-1
a = -L; b = L; % end points of grid
legendre_ord = 100;
basisq = cell(1,legendre_ord);
for i=0:legendre_ord-1
   basisq{i+1}   = @(x) legendreP(i,(2*x-(a+b))/(b-a));
   visualize = false;
   if visualize
       plot((-L:0.01:L)',basisq{i+1}((-L:0.01:L)'))
       hold on;
   end
end
% build data, of size (k x legendre_ord), rows enumerate
% collocation points, cols enumerate basis functions
Phi = zeros([k,legendre_ord]);
for i = 1:legendre_ord
    % evaluate phi_j at collocation points
    Phi(:,i) = basisq{i}(X);
end
% evaluate original function f
f_exact = f(X);

% find Moore-Penrose left psedoinverse
coeffs = (Phi*Phi')\f_exact;
coeffs = Phi'*coeffs;

norm(Phi*coeffs-f_exact)
%figure(2)
%plot(X,Phi*coeffs)

% test overall error by symbolic expression
x = sym('x');
f_exact_sym = f(x);
f_basis_sym = 0;
for i = 1:legendre_ord
    f_basis_sym = f_basis_sym + coeffs(i)*basisq{i}(x);
end
% absolute error expression
abs_err_sym = matlabFunction(log10(abs(f_exact_sym-f_basis_sym)));
% plot error
plot((-L:0.01:L)',abs_err_sym((-L:0.01:L)'),'Color','red','LineWidth',1.5);
%% Test 4(a) check L^2 orthogonality by Gauss-Legendre integration
clear; clc; rng('default');
% generate Gauss-Legendre collocation points on [-1,1]
L = 1;
% number of grid points
k = 200;
[X,W] = lgwt(k,-L,L);

% get two different Legendre polynomials
m = 3; n = 3;
basis1 = @(x) legendreP(m,x);
basis2 = @(x) legendreP(n,x);
if m == n
    disp(strcat(["> m = n, normalization = (1/m+0.5)", num2str( (1/(m+(1/2))))]))
end
% see MATLAB doc, m = n integration yields 1/(n+0.5) not 1
int_approx = sum(W.*basis1(X).*basis2(X));
disp(strcat(["> Numerical Integration Abs. Error: ", ...
    num2str(abs(int_approx))]));

% generate Gauss-Legendre collocation points on general [a,b]
a = -4; b = 5;
[X,W] = lgwt(k,a,b);
% get two different Legendre polynomials, rescaled
m = 3; n = 3;
if m == n
    disp(strcat(["> m = n, normalization = ((b-a)/2)*(1/m+0.5)", ...
        num2str( ((b-a)/2)*(1/(m+(1/2))))]))
end
basis1 = @(x) legendreP(m,(2*x-a-b)/(b-a));
basis2 = @(x) legendreP(n,(2*x-a-b)/(b-a));
int_approx = sum(W.*basis1(X).*basis2(X));
disp(strcat(["> (Scaled) Numerical Abs. Error: ", ...
    num2str(abs(int_approx))]));
%% Test 4(b): Testing builtin Legendre polynomials (normalized)
clear; clc; rng('default');
L = 1; % first [-1, 1]
k = 100; % number of grid points
a = -L; b = L;
[X,W] = lgwt(k,a,b);
m = 10; n = 10;
% MATLAB ref: http://matlab.izmiran.ru/help/techdoc/ref/legendre.html

% Warning: BASIS1, BASIS2 only take in vector inputs
basis1 = @(x) subsref( legendre(m,x,'norm'), ...
    struct('type','()','subs',{{1, ':'}})).';
basis2 = @(x) subsref( legendre(n,x,'norm'), ...
    struct('type','()','subs',{{1, ':'}})).'; % subsref used to only select
                                            % order 0 outputs from Legendre
% test integration
l2_iprod = sum(W.*basis1(X).*basis2(X));
if m == n
    disp(strcat("> m = n, L2 inner product = ", ...
        num2str(l2_iprod)))
else
    disp(strcat("> m not equal to n, L2 inner product = ", ...
        num2str(l2_iprod)))
end

% test rescaled
a = -4; b = 5;
adjust_factor = (b-a)/2;
disp(strcat("> Testing rescaled Legendre, adjustment factor = ", ...
    num2str(adjust_factor)));
[X,W] = lgwt(k,a,b);
m = 8; n = 1;
% Warning: BASIS1, BASIS2 only take in vector inputs
basis1 = @(x) subsref( legendre(m,(2*x-a-b)/(b-a),'norm'), ...
    struct('type','()','subs',{{1, ':'}})).';
basis2 = @(x) subsref( legendre(n,(2*x-a-b)/(b-a),'norm'), ...
    struct('type','()','subs',{{1, ':'}})).'; % subsref used to only select
                                            % order 0 outputs from Legendre
% test integration
l2_iprod = sum(W.*basis1(X).*basis2(X));
if m == n
    disp(strcat("> m = n, L2 inner product = ", ...
        num2str(l2_iprod)))
else
    disp(strcat("> m not equal to n, L2 inner product = ", ...
        num2str(l2_iprod)))
end                                        
                                            
                                            
%% Test 4(c): Expand a univariate function continuously with Normalized 
% Legendre basis
clear; clc; rng('default');
% number of grid points
k = 100;
% [-L,L]
a = -1; b = 10;
adjust_factor = (b-a)/2;
[X,W] = lgwt(k,a,b);
% exact function to approximate
f = @(x) exp(-x);
%f = @(x) exp(-x) + 10*sin(3*x.^2);
%f = @(x) exp(-0.25 * (x-2*sin(x.^2)).^2);
figure(1); 
plot((X), (f(X)), 'Color', 'red', 'LineWidth', 2);
title("Exact Function"); xlabel('x'); ylabel("f(x)"); hold on;
%f = @(x) sin(x.^2)+exp(-x);
% begin decomposition, max order of basis functions
legendre_ord = 10; 
basisq = cell(1,legendre_ord);
for i=0:legendre_ord-1
    % normalized Legendre basis
    basisq{i+1} = @(x) subsref( legendre(i,(2*x-(a+b))/(b-a),'norm'), ...
        struct('type','()','subs',{{1, ':'}})).';
   visualize = false;
   if visualize
       figure(3);
       plot((-L:0.01:L)',basisq{i+1}((-L:0.01:L)'))
       hold on;
   end
end

% find matrix A: A*f_exact(X) = a, coefficients of expansion
f_exact = f(X);
A = zeros([legendre_ord, k]);
for i = 1:legendre_ord
    A(i,:) = (1/adjust_factor)*W.*basisq{i}(X);
end
coeffs = A*f_exact;
% recover expansion
f_approx = 0;
for i = 1:legendre_ord
    f_approx = f_approx + coeffs(i)*basisq{i}(X);
end
% plot expansion with Legendre points
figure(2);
plot((X), (f(X)), 'Color', 'red', 'LineWidth', 2); hold on;
plot(X, f_approx, 'Color', 'blue', 'LineWidth', 2);
title(strcat("Exact vs. Approximate with Number of Legendre Basis: ", ...
    num2str(legendre_ord))); xlabel('x'); ylabel("f_approx(x)");

% plot abs. error with general points (to gauge interpolative property)
X = 0.01:0.01:10;
f_exact = f(X);
f_approx = 0;
for i = 1:legendre_ord
    f_approx = f_approx + coeffs(i)*basisq{i}(X);
end
figure(3); 
plot(X, abs(f_exact-f_approx.'), 'Color', 'black', 'LineWidth', 2);

%% Test 5: Legendre expansion of 2d function
clear; clc; rng('default');

% first approximate a 2d function
% this function is exactly decomposable
f = @(x,y) sin(x.^2).*cos(y.^2);
%f = @(x,y) sqrt(1./(x.^2 + y.^2));
%f = @(x,y) sin((x.^2)+(y.^2));
%f = @(x,y) y ./ x;
% fast decaying Gaussian
Mu = [-1,2]; Sigma = randn([2,2]); 
%f = @(x,y) (1/(2*pi))*...
%    exp(-0.5*( (x-Mu(1)).^2 + (y-Mu(2)).^2 ));
%f = @(x,y) (sin(x.^2) + cos(y.^2));

% create grid [-L,L]x[-L,L]
L = 10;
a = -L; b = L;
all_k = 50; % number of grid points in each dimension
all_legendre_ord = 5:80; % number of legendre orders 
cont_tt_err = zeros([1,length(all_k)]);
cont_tt_err_legendre = zeros([1, length(all_legendre_ord)]);
for ii = 1:length(all_k)
    ii
    k = all_k(ii);
    for jj = 1:length(all_legendre_ord)
        jj
        visualize = true;
        figure(1);
        if visualize
            subplot(2,2,1);
            legendre_ord = all_legendre_ord(jj);
            k = legendre_ord;
            [grid_1d,W] = lgwt(k,a,b);
            [X,Y] = meshgrid(grid_1d);
            f_exact = f(X,Y);
            surf(X,Y,f_exact); ...
                title("Exact Function $$f(x,y)$$",'interpreter',...
                'latex');
        end
        adjust_factor = (b-a)/2;
        % begin decomposition, max order of basis functions for each dim
        basisq = cell(1,legendre_ord);
        for i=0:legendre_ord-1
            % normalized Legendre basis
            basisq{i+1} = @(x) subsref( legendre(i,...
                (2*x-(a+b))/(b-a),'norm'), ...
                struct('type','()','subs',{{1, ':'}})).';
           visualize = true;
           if visualize
               subplot(2,2,2);
               plot((-L:0.01:L)',basisq{i+1}((-L:0.01:L)'))
               title(strcat("Number of Legendre Basis Used: ", ...
                   num2str(legendre_ord)), 'interpreter', 'latex')
               hold on;
           end
        end
        % find matrix A: A*f_exact(X) = a, coefficients of expansion
        A = zeros([legendre_ord, k]);
        for i = 1:legendre_ord
            A(i,:) = (1/adjust_factor)*W.*basisq{i}(grid_1d);
        end

        % TT cross f to compute the continuous TT
        U = sym('x', [1,2], 'real');
        u_sym_temp = U(:);
        u_sym_temp = u_sym_temp';
        f_ = matlabFunction(f(u_sym_temp(1), u_sym_temp(2)), ...
            'Vars', {u_sym_temp});
        tmp = cell(1,2); tmp{1} = tt_tensor(grid_1d); tmp{2} = ...
            tt_tensor(grid_1d);
        tt_grid = tt_meshgrid_vert(tmp(:));
        f_tt = amen_cross_s(tt_grid, f_, 1e-10, 'nswp', 15);
        if visualize
            % reshape f_tt back to matrix
            f_tt_reshaped = reshape(full(f_tt), [k k]);
            % surface plot
            subplot(2,2,3);
            surf(X,Y,f_tt_reshaped');
            title("Amen Cross $$f(x,y)$$", 'interpreter', 'latex');
            shading interp;
        end

        % get cores
        tt_cores = core(f_tt);
        core1 = tt_cores{1};
        core2 = tt_cores{2}';
        C1 = tns_mult(core1,1,A,2).';
        C2 = tns_mult(core2,2,A,2);
        tmp = cell([2,1]);
        tmp{1} = reshape(C1, [1 size(C1)]);
        tmp{2} = C2;
        coeff_tt = cell2core(tt_tensor, tmp);
        % check continuous TT is correct
        f_cont_tt = zeros([k k]);
        % flatten the grid to 1d vector for fast evaluation
        x_grid_points = grid_1d(:);
        y_grid_points = grid_1d(:);
        % build legendre data matrix
        basis_x = zeros([length(x_grid_points),legendre_ord]);
        basis_y = zeros([length(y_grid_points),legendre_ord]);
        for l = 1:legendre_ord
            % evaluate basis function at (x,y)
            basis_x(:,l) = basisq{l}(x_grid_points);
            basis_y(:,l) = basisq{l}(y_grid_points);
        end
        % contract with coefficients to find f(x,y)
        data_core1 = tns_mult(basis_x,2,C1,1);
        data_core2 = tns_mult(C2,2,basis_y,2);
        f_cont_tt = data_core1 * data_core2;

        %for i = 1:k
        %    for j = 1:k
        %        x1 = grid_1d(i);
        %        x2 = grid_1d(j);
        %        % evaluate (x,y) on bases
        %        basis_x = zeros([1,legendre_ord]);
        %        basis_y = zeros([1,legendre_ord]);
        %        for l = 1:legendre_ord
        %            basis_x(l) = basisq{l}(x1);
        %            basis_y(l) = basisq{l}(x2);
        %        end
        %        % contract with coefficient to find f(x,y)
        %        f_cont_tt(i,j) = tns_mult(C1,1,basis_x,2).'...
        %            *tns_mult(C2,2,basis_y,2);
        %    end
        %end

        if visualize
            % plot recovered function from continuous TT
            subplot(2,2,4);
            surf(X,Y,f_cont_tt'); 
            title("Continuous TT $$f(x,y)$$", 'interpreter', 'latex');
            shading interp;
        end
        tmp = f_cont_tt';
        % compute error in Frobenius norm
        norm(tmp(:)-f_exact(:))
        cont_tt_err_legendre(jj) = norm(tmp(:)-f_exact(:));
    end
    cont_tt_err(ii) = norm(tmp(:)-f_exact(:));
end
%% Test 6: General continuous TT evaluation for multivariate functions
clear; clc; rng("default");
% assume domain is [-L,L]
L = 3;
a = -L; b = L;
% exact function of 3 variables
%f = @(x,y,z) 1./(x.^2 + y.^2 + z.^2);
%f = @(x,y,z) mvnpdf([x y z]);
%f = @(x,y,z) exp(-(x.^2 + y.^2 + z.^2));
%f = @(x,y,z) sin(x.^2).^2.*cos(y.^2).^2;
f = @(x,y,z) mvnpdf([x,y,z]);

k = 50; % number of Gauss-Legendre points
[grid_1d,W] = lgwt(k,a,b);

% sqrt f
sqrt_f = @(x,y,z) sqrt(f(x,y,z));

% amen_cross discrete tensor
U = sym('x', [1,3], 'real');
u_sym_temp = U(:);
u_sym_temp = u_sym_temp';
f_ = matlabFunction(sqrt_f(u_sym_temp(1), u_sym_temp(2), u_sym_temp(3)), ...
    'Vars', {u_sym_temp});
tmp = cell(3,1); 
for i = 1:length(tmp)
    tmp{i} = tt_tensor(grid_1d);
end
% TT meshgrid
tt_grid = tt_meshgrid_vert(tmp);
% amen_cross sqrt(f)
sqrt_f_tt = amen_cross_s(tt_grid, f_, 1e-10, 'nswp', 30);

% exact tensor to compare to (not possible in high-dimensions)
[X,Y,Z] = meshgrid(grid_1d);

% build legendre transformation
legendre_ord = 50; % tune legendre orders
adjust_factor = (b-a)/2; % constant adjustment for [-L,L] domain
basisq = cell(1,legendre_ord);
for i = 0:legendre_ord-1
    % normalized Legendre basis
    basisq{i+1} = @(x) subsref( legendre(i,...
                (2*x-(a+b))/(b-a),'norm'), ...
                struct('type','()','subs',{{1, ':'}})).';
    visualize = false;
    if visualize
        figure(1);
        plot((-L:0.01:L)',basisq{i+1}((-L:0.01:L)'))
        title(strcat("Number of Legendre Basis Used: ", ...
           num2str(legendre_ord)), 'interpreter', 'latex')
        hold on;
    end
end

% find transformation matrix A: A*f_exact(X) = a, coefficients of expansion
A = zeros([legendre_ord, k]);
for i = 1:legendre_ord
    A(i,:) = (1/adjust_factor)*W.*basisq{i}(grid_1d);
end

% contract with tt_cross to obtain coefficient train
sqrt_f_tt_cores = core(sqrt_f_tt);
% preallocate cell array to store coefficient cores
coeff_cores = cell([length(sqrt_f_tt_cores),1]);
for i = 1:length(sqrt_f_tt_cores)
    % get sqrt_f core
    core_i = sqrt_f_tt_cores{i};
    % contract to get coefficient core
    C_i = tns_mult(A,2,core_i,1);
    if i ~= 1
        % need to reshape (for TT-Toolbox, cell2core)
        coeff_cores{i} = permute(C_i, [2 1 3]);
    else
        % no need to reshape the first core
        % just need to pad first dimension
        coeff_cores{i} = reshape(C_i, [1 size(C_i)]);
    end
end

% get coefficient train for sqrt(f)
coeff_tt = cell2core(tt_tensor,coeff_cores);
save_norm = norm(coeff_tt);
%coeff_tt = coeff_tt./norm(coeff_tt);
coeff_tt = qr(coeff_tt,"RL") * save_norm;
coeff_cores = core(coeff_tt);
coeff_cores{1} = reshape(coeff_cores{1}, [1 size(coeff_cores{1})]);
coeff_cores{2} = permute(coeff_cores{2},[2 1 3]);
coeff_cores{3} = permute(coeff_cores{3},[2 1 3]);


% compare continuous TT with exact tensor

% to build evaluated data TT, loop over
% coefficient cores and contract with basis evaluated on
% collocation points
f_approx_tt = cell([length(coeff_cores),1]);
for i = 1:length(f_approx_tt)
    % get coefficient core
    C_i = coeff_cores{i};
    % build basis data for this dimension
    basis_x_i = zeros([k,legendre_ord]);
    % for each basis function, evaluate on collocation grid
    for j = 1:legendre_ord
        % evaluate phi_j(X)
        basis_x_i(:,j) = basisq{j}(grid_1d(:));
    end
    % for this dimension, contract with coefficient tensor
    f_approx_tt{i} = permute(tns_mult(C_i,2,basis_x_i,2),[1 3 2]);
end

% last core should not be reshaped, permute back
f_approx_tt{end} = squeeze(f_approx_tt{end});
% tensor train sqrt_f_approx
sqrt_f_approx_tt = cell2core(tt_tensor, f_approx_tt);
% f_approx_tt should be a good approximation to sqrt(f)
%sqrt_f_exact = sqrt_f(Y,X,Z);
sqrt_f_exact = reshape(sqrt_f(Y(:),X(:),Z(:)), [k k k]);

norm(sqrt_f_approx_tt)
norm(sqrt_f_exact(:))
norm(full(sqrt_f_approx_tt)-sqrt_f_exact(:))

% squared TT
f_approx_tt = sqrt_f_approx_tt.^2;
%f_exact = f(Y,X,Z);
f_exact = reshape(f(Y(:),X(:),Z(:)), [k k k])
norm(f_approx_tt)
norm(f_exact(:))
norm(full(f_approx_tt)-f_exact(:))

% plot (x,y) slices varying z
[X,Y] = meshgrid(grid_1d);
for i = 1:k
    i
    figure(3);
    % get exact z-slice
    exact_f_xy = squeeze(f_exact(:,:,i));
    % surface plot of z-slice
    subplot(1,2,1);
    surf(X,Y,exact_f_xy);
    hold on; shading interp;
    
    % get approximated z-slice
    approx_f_xy = reshape(full(f_approx_tt(:,:,i)), [k k]);
    subplot(1,2,2);
    surf(X,Y,approx_f_xy);
    hold on; shading interp;
    pause(0.1);
    
    norm(full(f_approx_tt)-f_exact(:))
end
%% Test 6: Sample from Continuous TT
assert(exist("f_approx_tt",'var'), "> Please run the previous section. ");

% Consider 3d standard normal distribution
% 1. sample from p_1(x_1)
% form PSD weight matrix B by self-contracting coefficient TT
coeff_cores = core(coeff_tt);
coeff_cores{2} = permute(coeff_cores{2}, [2 1 3]);
coeff_cores{3} = permute(coeff_cores{3}, [2 1 3]);
B = tns_mult(coeff_cores{1},2,coeff_cores{2},1);
B = tns_mult(B,3,coeff_cores{3},1);
B = tns_mult(B,[2,3],B,[2,3]);

% number of samples from the first marginal
M = 5000;

% discretize a 1d grid (representing possible values in space)
space_grid = linspace(-L,L,1000);
marginal_prob = zeros(1,length(space_grid));
for i = 1:length(space_grid)
    % spatial points
    x_i = space_grid(i);
    % evaluate phi(x_i) for all basis
    phi_vec_x_i = zeros([1,legendre_ord]);
    for l = 1:legendre_ord
        phi_vec_x_i(l) = basisq{l}(x_i);
    end
    % evaluate p(x_i|x2,x3) by contraction 
    marginal_prob(i) = phi_vec_x_i * B * phi_vec_x_i';
end

% sample from the discrete probability vector
samples_x1 = randsample(space_grid,M,true,marginal_prob);

% evaluate marginal_prob on samples
samples_x1_prob = zeros(1,M);
for i = 1:M
    sample_x1 = samples_x1(i);
    % evaluate phi(x_i) for all basis
    phi_vec_x_i = zeros([1,legendre_ord]);
    for l = 1:legendre_ord
        phi_vec_x_i(l) = basisq{l}(sample_x1);
    end
    % evaluate p(x_i|x2,x3) by contraction 
    samples_x1_prob(i) = phi_vec_x_i * B * phi_vec_x_i';
end
% build the second marginal given samples_x1
% for each individual sample x1, the marginal distribution for x2 is
% different
for i = 1:length(samples_x1)
    % samples x1
    x1 = samples_x1(i);
    % evaluate phi(x_i) for all basis
    phi_vec_x_i = zeros([1,legendre_ord]);
    for l = 1:legendre_ord
        phi_vec_x_i(l) = basisq{l}(x1);
    end
    % weight matrix M
    M_x1 = phi_vec_x_i.*phi_vec_x_i';
    
end
%% 

% Test 7: testing custom legendre function
clear; clc; rng('default');
[x,w] = lgwt(100,-1,1);
p = 10; % highest order of legendre polynomials
% generate legendre polynomials evaluated at GL points
y = get_legendre(x,p,true);
% numerical integration (results stored in p x p matrix)
integral_result = zeros(p+1,p+1);
for i = 1:p+1
    for j = 1:p+1
        % numerically integrate
        integral_result(i,j) = sum(w.*y(:,i).*y(:,j));
    end
end
disp(strcat("> norm difference from identity = ", ...
    num2str(norm(integral_result-eye(p+1)))));

% test scaled [-1,1] grid
scaling = 3;
L = scaling;
[x,w] = lgwt(100,-L,L);
adjust_factor = (L-(-L))/2;
p = 10;
y = get_legendre((2*x-(-L+L))./(L-(-L)),p,true);
integral_result = zeros(p+1,p+1);
for i = 1:p+1
    for j = 1:p+1
        % numerically integrate
        integral_result(i,j) = (1/adjust_factor)*sum(w.*y(:,i).*y(:,j));
    end
end
disp(strcat("> norm difference from identity = ", ...
    num2str(norm(integral_result-eye(p+1)))));

% test shifted grid [a,b]
a = -10.98911; b = 2.5898;
[x,w] = lgwt(100,a,b);
adjust_factor = (b-a)/2;
p = 10;
y = get_legendre((2*x-(a+b))./(b-a),p,true);
integral_result = zeros(p+1,p+1);
for i = 1:p+1
    for j = 1:p+1
        % numerically integrate
        integral_result(i,j) = (1/adjust_factor)*sum(w.*y(:,i).*y(:,j));
    end
end
disp(strcat("> norm difference from identity = ", ...
    num2str(norm(integral_result-eye(p+1)))));

%% Test 8: Comparison of Design matrices (legendreP versus custom)
clear; clc; rng('default');
% build custom design matrix
p = 10;
n=p+1;
n = 1000;
[xgl,wgl] = lgwt(n,-1,1);
Agl = get_legendre(xgl,p,true);
michael_G = pinv(Agl);

% build design matrix using legendreP
% number of grid points
k = n;
% [-L,L]
a = -4; b = 4;
adjust_factor = (b-a)/2;
[X,W] = lgwt(k,a,b);

% begin decomposition, max order of basis functions
legendre_ord = p+1; 
basisq = cell(1,legendre_ord);
for i=0:legendre_ord-1
    % normalized Legendre basis
    basisq{i+1} = @(x) subsref( legendre(i,(2*x-(a+b))/(b-a),'norm'), ...
        struct('type','()','subs',{{1, ':'}})).';
end

% form design matrix
A = zeros([legendre_ord, k]);
for ord = 1:legendre_ord
    A(ord,:) = basisq{ord}(X);
end
A = A';
norm(A-Agl)

% find matrix A: A*f_exact(X) = a, coefficients of expansion
bob_G = zeros([legendre_ord, k]);
for i = 1:legendre_ord
    bob_G(i,:) = (1/adjust_factor)*W.*basisq{i}(X);
end
norm(bob_G-michael_G)

%% Comparison of coefficients obtained in 2 ways

    % there are two ways to obtain coefficients such that
    %       D * c = f
    %
    % 1. directly invert design matrix
    % 2. use orthogonality condition
    %
    %
clear; clc; rng('default');
M = 1000; % number of legendre polynomials (+1 because degree starts from 0)
N = 2^5; % number of nodes
a = -1;
b = 1;
[grid_x,W] = lgwt(N,a,b);
R = 200;

% simple 1d function
f = @(x) exp(-0.5 * (R*x).^2);
% f evaluated at grid points
f_eval = f(grid_x);
% create design matrix
D = get_legendre(grid_x,M,true);
% inverting design matrix 
c = pinv(D)*f_eval;

% create matrix by orthogonality condition
D_tilde = D';
for i = 1:N
    D_tilde(:,i) = (1/(sqrt((b-a)/2))) * W(i).*D_tilde(:,i);
end
c_tilde = D_tilde*f_eval;
norm(D*c - f_eval)
norm(D*c_tilde - f_eval)




