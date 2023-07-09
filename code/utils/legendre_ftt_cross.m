function [coeff_tt, basisq] = ...
    legendre_ftt_cross_copy(L, k, d, sqrt_f, legendre_ord, tol)

%       (Uniform grid)
%       Functional expansion using Legendre basis. inp us same as
%       the argument requirement for amen_cross, a cell array of
%       size 1 x d, each cell containing a compressed grid in 1d.
%
%       Returns coefficient train and the basis used. Mode sizes
%       of coefficient train should match the number of basis used
%       for that dimension. SQRT_F should be made the required format
%       for amen_cross. The grid generated is Gauss-Legendre collocation.
%       tol is tolerance for amen_cross
%       points

%       Dependency: tt_tensor/amen_cross.m, tt_tensor/qr.m, lgwt.m


% left and right end points on spatial grid
a = -L; b = L;
[spatial_grid,W] = lgwt(k,a,b);
% grid for amen_cross
grid_tt = cell([d,1]);
X_tt = tt_tensor(spatial_grid);
for i = 1:d
    grid_tt{i} = X_tt;
end

% TT meshgrid
tt_grid = tt_meshgrid_vert(grid_tt);
% amen_cross sqrt(f)
sqrt_f_tt = amen_cross_s(tt_grid, sqrt_f, tol,'nswp',100,'kickrank',10);

% build legendre transformation
adjust_factor = (b-a)/2; % constant adjustment for [-L,L] domain
basisq = cell([legendre_ord,1]);
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
    A(i,:) = (1/adjust_factor)*W.*basisq{i}(spatial_grid);
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

% renormalize and put coefficient train in Right-Left QR form
% The QR form will implicitly orthonormalize the TT, need
% to multiply back the norm
%norm(coeff_tt)
coeff_tt_norm = norm(coeff_tt);
coeff_tt = qr(coeff_tt, "RL") * coeff_tt_norm;
end