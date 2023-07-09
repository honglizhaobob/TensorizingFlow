function [coeff_tt, basisq] = ...
    legendre_ftt_cross(inp, sqrt_f, legendre_ord, tol)

%       (General d-dimensional grid)
%       Functional expansion using Legendre basis. inp is meta
%       data for grid in each dimension. Each row of inp should
%       specify [a_i, b_i, k_i] where (a_i, b_i) is domain, k_i
%       is number of grid points in that dimension.

%
%       Returns coefficient train defined on [a, b] (general domain)
%       and the basis used (also defined on [a, b]). Mode sizes
%       of coefficient train should match the number of basis used
%       for that dimension. SQRT_F should be made the required format
%       for amen_cross. The grid generated is Gauss-Legendre collocation.
%       tol is tolerance for amen_cross
%       points

%       Dependency: tt_tensor/amen_cross.m, tt_tensor/qr.m, lgwt.m

% grid metadata: column 1 are left end points, column 2 right end points
% column 3 is number of grid points for 1d grid.
assert(size(inp, 2) == 3, "> Must use correct metadata format. ")
d = size(inp,1); 
grid_metadata = inp;

% get Gauss-Legendre points in each dimension

% grid for initial amen_cross
grid_tt = cell([d,1]); 
% store all Gauss-Legendre weights 
all_spatial_grids = cell([d,1]);
% ======= old code 
all_weights = cell([d,1]);
% =======
for i = 1:d
    metadata_i = grid_metadata(i,:);
    a_i = metadata_i(1); 
    b_i = metadata_i(2);
    k_i = metadata_i(3);
    [spatial_grid_i, W_i] = lgwt(k_i,a_i,b_i);
    % ======= old code
    all_weights{i} = W_i;
    % =======
    all_spatial_grids{i} = spatial_grid_i;
    grid_tt{i} = tt_tensor(spatial_grid_i);
end

% create TT meshgrid
tt_grid = tt_meshgrid_vert(grid_tt);

% amen_cross sqrt(f)
sqrt_f_tt = amen_cross_s(tt_grid, sqrt_f, tol, 'nswp', 200);
% === testing
%sqrt_f_tt_visualize = full(sqrt_f_tt,[length(spatial_grid_i),...
%    length(spatial_grid_i)]);
%min(min(sqrt_f_tt_visualize))
% ===
% norm(sqrt_f_tt)

% build legendre transformation
% all adjustment factors (to ensure orthonormal)
%all_adjust_factors = (1/2).*(grid_metadata(:,2)-grid_metadata(:,1));

% create all basis functions in each dimension
% use the same number of basis functions in each
% dimension.
basisq = cell([d,legendre_ord]);
% each dimension adjusted differently
for dim = 1:d
    % normalized Legendre basis
    a_i = inp(dim,1);
    b_i = inp(dim,2);
    for i = 0:legendre_ord-1
        % shift [a_i,b_i] -> [-1,1]
        basisq{dim,i+1} = @(x) subsref( legendre(i,...
                    (2*x-(a_i+b_i))/(b_i-a_i),'norm'), ...
                    struct('type','()','subs',{{1, ':'}})).';
    end
end

% find transformation matrix A for each dimension
A = cell([d,1]);
for dim = 1:d
    % for each dimension we store a design matrix A_i
    % ====== new, uses direct pseudo-inverse of design matrix
    
    % grid points used for interpolation in this dimension
    spatial_grid_i = all_spatial_grids{dim};
    D_i = get_legendre(spatial_grid_i,legendre_ord-1,true);
    % A_i is the pseudoinverse of the design matrix D_i
    A_i = pinv(D_i);
    
    
    % ======
    
    % ====== old, uses orthogonality condition, which is probably not
    % correct (see test in cont_tt_experiments.m)
    %A_i = zeros(legendre_ord,k_i);
    %adjust_factor_i = all_adjust_factors(dim);
    %W_i = all_weights{dim};
    %spatial_grid_i = all_spatial_grids{dim};
    %for i = 1:legendre_ord
    %   % A_i(i,:) = ...
    %   %     (1/(sqrt(adjust_factor_i)))*W_i.*basisq{dim,i}(spatial_grid_i);
    %    A_i(i,:) = ...
    %            W_i.*basisq{dim,i}(spatial_grid_i);
    %end
    % ====== old code above
    A{dim} = A_i;
end


% contract with tt_cross to obtain coefficient train
sqrt_f_tt_cores = core(sqrt_f_tt);
% preallocate cell array to store coefficient cores
coeff_cores = cell([length(sqrt_f_tt_cores),1]);
for i = 1:d
    % get sqrt_f core
    core_i = sqrt_f_tt_cores{i};
    % contract to get coefficient core
    % contract with coefficient matrix A_i
    A_i = A{i};
    C_i = tns_mult(A_i,2,core_i,1);
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
% ==== check
%sqrt_f_tt = full(sqrt_f_tt,[sqrt_f_tt.n']);
%D = sqrt_f_tt;
%U = get_legendre(spatial_grid_i,legendre_ord-1,true);
%B2 = U'*diag(W_i)*D*diag(W_i)*U;
%norm(D - U*B2*U','fro');
%B1 = full(coeff_tt,[coeff_tt.n(1),coeff_tt.n(1)]);
%norm(B1-B2,'fro');
%min(min( U*B2*U'));
%D2 = approx_density(coeff_tt,{spatial_grid_i;spatial_grid_i});
%D2 = full(D2,[D2.n']);
%min(min(D2))

% ====

% renormalize and put coefficient train in Right-Left QR form
% The QR form will implicitly orthonormalize the TT, need
% to multiply back the norm
coeff_tt_norm = norm(coeff_tt);
coeff_tt_cores = core(coeff_tt);

coeff_tt_cores = RLqr(coeff_tt_cores);
coeff_tt_cores{1} = reshape(coeff_tt_cores{1},[1,size(coeff_tt_cores{1})]);
coeff_tt_new = cell2core(tt_tensor(),coeff_tt_cores);
%disp("===crossed")
%norm(coeff_tt_new - coeff_tt)

%coeff_tt = qr(coeff_tt, "RL") * coeff_tt_norm;
% ====
% ==== check
%D2 = approx_density(coeff_tt,{spatial_grid_i;spatial_grid_i});
%D2 = full(D2,[D2.n']);
%min(min(D2))

%coeff_tt = qr(coeff_tt, "RL") * coeff_tt_norm;
%
% ====
coeff_tt = coeff_tt_new;
end