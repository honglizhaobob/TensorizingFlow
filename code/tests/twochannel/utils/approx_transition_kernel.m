function transition_tt = approx_transition_kernel(grid, N, tol, dt, beta)
    % Use interpolative decomposition to factorize nonequilibrium path kernel
    % into TT cores
    % Given N discretization points, the joint density is assumed to be
    % expressed as a product of "kernels" with inputs of R^2
    %
    % p(x1, x2, ..., xN; x0=xA, xN+1=xB) = 
    % K(xA, x1) * K(x1, x2) * ... * K(xN, xB)
    % - for N discretization points, the result will be 2*N cores, with
    % each core representing a coordinate of a point, e.g. y coordinate
    % of \mathbb{R}^2 random variable X_i. Instead, we build 2*(N+1) cores
    % (treating xA, xB as free variables), then we locate and "index out"
    % xA and xB. Effectively, the time interval will have length 
    % ( grid(end)-grid(1) ) / ( (N+2)-1 )
    % (counting xA, xB as start and terminal points, there are (N+1)
    % intervals).
    %
    % ========== Notes ==========
    % 1. K can be considered as K^2 x K^2 matrices with column major 
    % stacking (MATLAB default).
    %
    % 2. Assumes a uniform GRID
    %
    % 3. Refer to [Gabrie] along with their code on github 
    % for the exact form of the "target path action"
    %
    % 4. As per experiments in svd_transition_energy_test.m
    % the variable ordering of the decomposition is 
    % ( xN_2,xN_1, ..., x1_2,x1_1 )
    %
    %
    % 5. Several parameters are static in this routine
    %     xA = [-1, 0]
    %     xB = [1, 0]
    %     c = 0 or 2.5
    % 6. For large grid and large N, there is a MATLAB memory constraint.
    % Within the routine we need to create a matrix of size k^2 x k^2
    % which will restrict our capacity.
    %
    %
    % ========== Dependencies ==========
    %           drift.m 
    %           flatten_2d.m
    %           isin_grid.m
    %
    % Input:
    %       grid,                   (1d array) grid used for space, should
    %                               be ordered X1 < X2 < ... < Xk
    %       N,                      (scalar)   number of R^2 variables in
    %                               the system
    %       tol,                    (scalar)   tolerance level for SVD 
    %                               truncation
    %       dt,                     (scalar)   parameter, time step size
    %       beta,                   (scalar)   parameter, inverse
    %                               temperature
    %
    %
    %
    % Output:
    %       transition_tt           (tt_tensor) tensor train with effective
    %                               dimension 2*N with boundary xA, xB 
    %                               considered
    if (N < 3)
        error("Input Error: Currently only supports N > 2");
    end
    % boundary points
    xA = [-1, 0]; xB = [1, 0];
    % nonconservative coefficient
    c = 2.5;
    %c = 0;
    k = length(grid);
    % end points
    x_start = grid(1);
    dx = grid(2)-grid(1);
    % boundary points must be in the grid (for indexing to work)
    assert(isin_grid(xA, grid) && isin_grid(xB, grid), ...
        "> Both boundary points xA, xB must be indexable in the grid. ");
    [~, ~, ~, row_coord, col_coord] = ...
        flatten2d(grid);
    % all possible values for each random X_i
    all_points = [row_coord(:), col_coord(:)];
    
    % form (k^2) x (k^2) matrix element-wise in space
    % once for row-coord, once for col-coord, get
    % (k^2 x k^2 x 2) array, which we compute Euclidean norm
    % along the last dimension to obtain (k^2 x k^2) sum of squares
    K_common = path_energy_kernel(all_points, beta, dt, c);
    K_xA = path_energy_kernel_xA(all_points, beta, dt, c);
    
    % ==========
    % Step 1, interpolative decomposition on 2d space
    % ==========
    % Due to path_energy_kernel.m the ordering of points is currently:
    % (X_N, X_N-1, ..., X_2, X_1)
    % We have p(X_i+1, X_i) = A(X_i+1)*B'(X_i)
    
    % perform SVD on K_common
    [A_r, B_r, r] = interp_decomp(K_common, tol);
    
    % perform SVD on K_xA
    [A_r_xA, B_r_xA, r_xA] = interp_decomp(K_xA, tol);
    
    % preallocate
    core = zeros(r, k^2, r);
    core_x1 = zeros(r, k^2, r_xA);
    
    % construct common core
    for i = 1:r
        for j = 1:r
            core(i,:,j) = A_r(:,i).*B_r(:,j);
        end
    end
    
    % construct core for x1
    for i = 1:r
        for j = 1:r_xA
            core_x1(i,:,j) = A_r(:,i).*B_r_xA(:,j);
        end
    end
    
    % form a tensor train of dimension N+2, treating 
    % xA, xB as varibles, which will be removed by indexing
    % at appropriate positions
    % i.e. modes = [xA, x1, x2, ..., xN, xB]
    % xA, xB: 1 x k^2 x r, r x k^2 x 1
    % rest  : r x k^2 x r
    % create cores (for x \in \R^2)
    left_boundary = reshape(B_r, [1 k^2, r]);
    right_boundary = reshape(A_r_xA', [r_xA k^2 1]);         
    common_core = core;                 
    % convert xA, xB into index positions 1:k^2
    [xA_xpos, xA_ypos] = find_idx2d(xA, grid);
    [xB_xpos, xB_ypos] = find_idx2d(xB, grid);
    % find xA and xB in 1:k^2, column major
    % now separate coordinates in each core
    % (r x k^2 x r --> (r x k x r) \otimes (r x k x r)
    % (1 x k^2 x r --> (1 x k x r) \otimes (r x k x r)
    % (r x k^2 x 1 --> (r x k x r) \otimes (r x k x 1)
    % after reshaping, will have a total of 2*N cores
    % one representing each coordinate of each variable
    
    % create 2N+4 cores, representing xi_1, xi_2 for i = 1,2,...,N+2
    % where the boundaries are tentatively treated as free variables
    core_cells = cell([2*(N+2), 1]);
    % loop over each core we have for R^2
    % decompose it into 2 cores, each for R
    % indexing 1:(N+2) --> (2:2*(N+2))-1 and (2:2*(N+2))
    for i = 1:(N+2)
        if i == 1
            % special case: xA
            core_i = left_boundary;
            %assert( all(size(core_i) == [1, k^2, 1]), ...
            %        "left boundary should have size [1 k^2 r]");
            % reshape into k x kr
            core_i = reshape(core_i, [1*k, k*r]);
            [a_r,b_r,r2] = interp_decomp(core_i, tol);
            core_cells{2*i-1} = reshape(a_r, [1 k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r]); 
            disp( strcat("===== error from left boundary SVD = ", ...
                num2str(norm(a_r*b_r'-core_i))) )
        % now core for X_1 is different (or second core before
        % decouping x, y)
        elseif i == N+1
            core_i = core_x1;
            %assert( all(size(core_i) == [r, k^2, r_xA]), ...
            %        "left boundary should have size [r k^2 r_xA]");
            % decouple x,y
            core_i = reshape(core_i, [r*k, k*r_xA]);
            [a_r,b_r,r2] = interp_decomp(core_i, tol);
            core_cells{2*i-1} = reshape(a_r, [r k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r_xA]); 
        %=====
        elseif i == N+2
            core_i = right_boundary;
            % special case: xB
            %assert( (size(core_i, 1) == r_xA) & ...
            %        (size(core_i, 2) == k^2), ...
            %        "right boundary should have size [r_xA k^2]");
            % reshape into rk x k
            core_i = reshape(core_i, [r_xA*k k*1]);
            % SVD to separate two coordinates
            [a_r,b_r,r2] = interp_decomp(core_i, tol);
            core_cells{2*i-1} = reshape(a_r, [r_xA k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k 1]);
            disp( strcat("===== error from right boundary SVD = ", ...
                num2str(norm(a_r*b_r'-core_i))) )
        else
            % do nothing (common core is moved outside of for loop)
        end
        
    end
    % common core (only if N >= 3)
    if N >= 3
        core_i = common_core;
        %assert( (size(core_i, 1) == r) & ...
        %        (size(core_i, 2) == k^2) & ...
        %        (size(core_i, 3) == r), ...
        %        "common core should have size [r k^2 r]");
        % reshape into rk x kr
        core_i = reshape(core_i, [r*k, k*r]);
        % SVD to separate two coordinates
        [a_r,b_r,r2] = interp_decomp(core_i, tol);
        disp( strcat("=== error from common core SVD: ", ...
            num2str(norm(a_r*b_r'-core_i))) )
        % only need to repeat common core
        for i = 2:N
            core_cells{2*i-1} = reshape(a_r, [r k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r]); 
        end
    end
    % after building all cores, return the TT
    transition_tt = cell2core(tt_tensor, core_cells);
    % index out xA and xB
    % needs to create an array indexer programmatically
    arr_indexer = cell([1, 2*(N+2)]); 
    for i = 1:length(arr_indexer)
        arr_indexer{i} = ':';
    end
    % ordering: xN_2, xN_1, xN-1_2, xN-1_1, ..., x1_2, x1_1, x0_2, x0_1
    arr_indexer{2*1-1} = xB_ypos; 
    arr_indexer{2*1}   = xB_xpos;
    arr_indexer{2*(N+2)-1} = xA_ypos;
    arr_indexer{2*(N+2)}   = xA_xpos;
    transition_tt = transition_tt(arr_indexer{:});
    assert( length(transition_tt.n) == 2*N, ...
        "indexing failed, needs to rid of 2 dimensions" );
end