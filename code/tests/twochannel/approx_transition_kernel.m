function transition_tt = approx_transition_kernel(grid, N, tol, dt, beta)
    % Use interpolative decomposition to factorize nonequilibrium path kernel
    % into TT cores
    % Factorizes p(x1, x2)*p(x2, x3) into tensor cores, where each
    % x1, x2, x3 \in \mathbb{R}^2
    % p(x1, x2) and p(x2, x3) can be considered as K^2 x K^2 matrices
    % with column major stacking (MATLAB default).
    
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
    %  
    
    assert(N >= 3, "currently only supports dim > 2");
    
    % for simplicity, assume all grids are the same
    % i.e. solving the problem in hypercube, with uniform 
    % step size
    k = length(grid); dx = grid(2)-grid(1);
    % boundary points
    xA = [-1, 0]; xB = [1, 0];
    % nonconservative coefficient
    c = 0;
    % check boundary points are included in the grid, otherwise
    % the TT routine will not work
    assert(ismember(xA(1), grid) & ismember(xA(2), grid), ...
        "boundary points need to be included in the grid"); 
    assert(ismember(xB(1), grid) & ismember(xB(2), grid), ...
        "boundary points need to be included in the grid"); 
    % end points
    x_start = grid(1); 
    x_end = grid(end);
    idx = 1:(k^2);
    % column major indexer into 2d grid [-L,L]x[-L,L]
    row_indexer = ceil(idx / k); 
    col_indexer = mod(idx-1, k) + 1;
    assert(all(row_indexer <= k) & all(row_indexer >= 1), ...
        "1 <= row index <= k");
    assert(all(col_indexer <= k) & all(row_indexer >= 1), ...
        "1 <= col index <= k");
    % convert indexer into coordinates
    row_coord = x_start + (row_indexer - 1) * dx;
    col_coord = x_start + (col_indexer - 1) * dx;
    % ==========
    % WARNING: k^2 may be prohibitively large and expensive
    % get all possible points, of shape (k^2 x 2)
    % use indexer, need to make sure that points on 2d grid
    % is enumerated in column major
    % ==========
    all_points = [row_coord(:), col_coord(:)];
    assert(all(size(all_points) == [k^2, 2]), ...
        "need to be size k^2 x 2");
    % all points for x_i-1, x_i, x_i+1 are all the same, since
    % it denotes possible values
    % compute drift
    all_drift = drift(all_points, c);
    
    % we form the (k^2) x (k^2) matrix element-wise in space
    % once for first-coord, once for second-coord, then we get
    % (k^2 x k^2 x 2) tensor, which we compute Euclidean norm
    % along the last dimension to obtain (k^2 x k^2)
    %
    % ( X_i+1 ) - ( X_i + \Delta t * b( X_i ) )
    energy_row = ...
        all_points(:,1).'...
            - (all_points(:,1) + dt * all_drift(:,1)); % size k^2 x k^2
    energy_col = ...
        all_points(:,2).'...
            -(all_points(:,2) + dt * all_drift(:,2)); % size k^2 x k^2
    
    % xA's core is different as it is
    % (x1-xA)/dt - b(x1)
    % = ( ( X_1 / dt ) - b(X_1) ) - ( X_A / dt )
    xA_energy_row = ...
        ( all_points(:,1) - dt * all_drift(:,1) ).' - ...
        all_points(:,1);
    xA_energy_col = ...
        ( all_points(:,2) - dt * all_drift(:,2) ).' - ...
        all_points(:,2);
    K_xA = exp( -0.25 .* ( beta/dt ) .* ...
        ( xA_energy_row.^2 + xA_energy_col.^2 ) );
    
    % perform SVD on K_xA
    [U_xA, S_xA, V_xA] = svd(K_xA);
    A_xA = U_xA*S_xA; B_xA = V_xA;
    idx = find(diag(S_xA)./max(diag(S_xA)) > tol);
    r_xA = length(idx);
    A_r_xA = A_xA(:,idx); B_r_xA = B_xA(:,idx);
    disp(strcat("===== error from xA kernel SVD = ", ...
        num2str(norm(A_r_xA*B_r_xA'-K_xA))));
    % need common core to combine into (rA x k^2 x r)
    
    % kernel matrix p(X_i-1, X_i) = 
    % exp(-beta/4/dt * norm(energy_row, energy_col))
    K = energy_row.^2 + energy_col.^2;                 % size k^2 x k^2
    K = ( beta/dt ) * K;
    K = exp(-0.25 * K);
    
    % form interpolative decomposition using SVD
    [U,S,V] = svd(K);
    A = U*S; B = V;                    % K = A * B'
    % truncate to first r columns: A is (k x r)
    idx = find( diag(S)/max(diag(S)) > tol );
    r = length(idx);
    A_r = A(:,idx);
    % truncate to first r columns: B is (k x r), B' is (r x k)
    B_r = B(:,idx);
    disp( strcat("===== error from kernel SVD = ", ...
        num2str(norm(A_r*B_r'-K))) ) 
    % preallocate core
    core = zeros(r, k^2, r);
    % 09/08/2021 -- added core x1, should be second core before decoupling
    % x,y, core should have size (r_xA, k^2, r), representing core 1
    % core xA has size (1, k^2, r_xA)
    % core 2 has size (r, k^2, r), common core
    core_x1 = zeros(r_xA, k^2, r);
    for i = 1:r_xA
        for j = 1:r
            core_x1(i,:,j) = A_r_xA(:,i).*B_r(:,j);
        end
    end
    
    for i = 1:r
        for j = 1:r
            core(i,:,j) = A_r(:,i).*B_r(:,j);
        end
    end
    % form a tensor train of dimension N+2, treating 
    % xA, xB as varibles, which will be removed by indexing
    % at appropriate positions
    % i.e. modes = [xA, x1, x2, ..., xN, xB]
    % xA, xB: 1 x k^2 x r, r x k^2 x 1
    % rest  : r x k^2 x r
    % create cores (for x \in \R^2)
    
    
    % (new, 09/08/2021) xA core is size (1, k^2, r_xA)
    % xB core remains the same
    %=====
    %xA_core = reshape(B_r, [1, k^2 r]);             % 1 x k^2 x r
    xA_core = reshape(B_r_xA, [1 k^2, r_xA]);
    %=====

    xB_core = reshape(A_r', [r k^2 1]);             % r x k^2 x 1
    common_core = core;                             % r x k^2 x r

    % convert xA, xB into index positions within the range 1:k^2
    xA_xpos = round((xA(1) - x_start)/dx) + 1;
    xA_ypos = round((xA(2) - x_start)/dx) + 1;
    xB_xpos = round((xB(1) - x_start)/dx) + 1;
    xB_ypos = round((xB(2) - x_start)/dx) + 1;
    assert(all([grid(xA_xpos), grid(xA_ypos)] == xA), ...
        "indices for xA are incorrect");
    assert(all([grid(xB_xpos), grid(xB_ypos)] == xB), ...
        "indices for xB are incorrect");
    % find xA and xB in 1:k^2, column major
    %xA_ind = sub2ind([k k], xA_ypos, xA_xpos); % MATLAB array got
    %transposed
    %xB_ind = sub2ind([k k], xB_ypos, xB_xpos);
    %assert(all( round(all_points(xA_ind, :)) == xA ) & ...
    %       all( round(all_points(xB_ind, :)) == xB ), ...
    %       "1:k^2 index location for xA, xB incorrect");
    

    % now separate coordinates in each core
    % (r x k^2 x r --> (r x k x r) \otimes (r x k x r)
    % (1 x k^2 x r --> (1 x k x r) \otimes (r x k x r)
    % (r x k^2 x 1 --> (r x k x r) \otimes (r x k x 1)
    
    % after reshaping, will have a total of 2*N cores
    % one representing each coordinate of each variable
    %core_cells = cell([2*N, 1]);
    
    % create 2N+4 cores, representing xi_1, xi_2 for i = 1,2,...,N+2
    % where the boundaries are tentatively treated as free variables
    core_cells = cell([2*(N+2), 1]);
    

    % loop over each core we have for R^2
    % decompose it into 2 cores, each for R
    % indexing 1:(N+2) --> (2:2*(N+2))-1 and (2:2*(N+2))
    for i = 1:(N+2)
        if i == 1
            % special case: xA
            core_i = xA_core;
            %==== (new, 09/08/21)
            %assert( all(size(core_i) == [1, k^2, r]), ...
            %        "left boundary should have size [1 k^2 r]");
            assert( all(size(core_i) == [1, k^2, r_xA]), ...
                    "left boundary should have size [1 k^2 r_xA]");
            %==== comment: to reflect changes in core K(xA, x1)
            
            
            % reshape into k x kr
            %core_i = reshape(core_i, [1*k, k*r]);
            %===== (new, 09/08/21)
            core_i = reshape(core_i, [1*k, k*r_xA]);
            %=====
            
            
            % SVD to separate two coordinates
            [u, s, v] = svd(core_i);
            a = u*s; b = v;                % A = (k x k*r), B = (k*r x k*r)
            idx = find( (diag(s)/max(diag(s))) > tol );
            r2 = length(idx);
            %===== new, 09/08/21
            % truncate, possibly with a different rank r2
            %a_r = a(:, idx); b_r = b(:, idx);   % A_r = (k x r2), 
                                                % B_r = (k*r x r2)
                                                % B_r' = (r2 x k*r)
                                                
            a_r = a(:, idx); b_r = b(:, idx);   % A_r = (k x r2), 
                                                % B_r = (k*r_xA x r2)
                                                % B_r' = (r2 x k*r_xA)
            core_cells{2*i-1} = reshape(a_r, [1 k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r_xA]); 
            %=====
            disp( strcat("===== error from left boundary SVD = ", ...
                num2str(norm(a_r*b_r'-core_i))) )
        %===== new, 09/08/21
        % new code, now core for X_1 is different (or second core before
        % decouping x, y)
        elseif i == 2
            core_i = core_x1;
            assert( all(size(core_i) == [r_xA, k^2, r]), ...
                    "left boundary should have size [r_xA k^2 r]");
            % decouple x,y
            core_i = reshape(core_i, [r_xA*k, k*r]);
            [u, s, v] = svd(core_i);
            a = u*s; b = v;                % A = (r_xA*k x r2), B = (r2 x k*r)
            idx = find( (diag(s)/max(diag(s))) > tol );
            r2 = length(idx);
            a_r = a(:, idx); b_r = b(:, idx);
            core_cells{2*i-1} = reshape(a_r, [r_xA k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r]); 
        %=====
        elseif i == N+2
            core_i = xB_core;
            % special case: xB
            assert( (size(core_i, 1) == r) & ...
                    (size(core_i, 2) == k^2), ...
                    "right boundary should have size [r k^2]");
            % reshape into rk x k
            core_i = reshape(core_i, [r*k k*1]);
            % SVD to separate two coordinates
            [u, s, v] = svd(core_i);
            a = u*s; b = v;             % A = (rk x k), B = (k x k)
            idx = find( (diag(s)/max(diag(s))) > tol );
            r2 = length(idx);
            % truncate, possibly with a different rank r3
            a_r = a(:, idx); b_r = b(:, idx); % A_r = (rk x r3)
                                              % B_r = (k x r3)
                                              % B_r' = (r3 x k)
            core_cells{2*i-1} = reshape(a_r, [r k r2]);
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
        assert( (size(core_i, 1) == r) & ...
                (size(core_i, 2) == k^2) & ...
                (size(core_i, 3) == r), ...
                "common core should have size [r k^2 r]");
        % reshape into rk x kr
        core_i = reshape(core_i, [r*k, k*r]);
        % SVD to separate two coordinates
        [u, s, v] = svd(core_i);
        a = u*s; b = v;                % A = (rk x kr), B = (rk x kr)
        idx = find( (diag(s)/max(diag(s))) > tol );
        r2 = length(idx);
        % truncate, possibly with a different rank r2, now r2 = r
        a_r = a(:, idx); b_r = b(:, idx);  % A_r = (rk x r2)
                                           % B_r = (r x kr)
                                           % B_r' = (kr x r2)
        disp( strcat("=== error from common core SVD: ", ...
            num2str(norm(a_r*b_r'-core_i))) )
        
        % only need to repeat common core
        % ===== new, 09/08/21
        % comment: now three cores: core_xA, core_x1, core_xB are special
        %for i = 2:(N+1)
        %    core_cells{2*i-1} = reshape(a_r, [r k r2]);
        %    core_cells{2*i}   = reshape(b_r', [r2 k r]); 
        %end
        for i = 3:(N+1)
            core_cells{2*i-1} = reshape(a_r, [r k r2]);
            core_cells{2*i}   = reshape(b_r', [r2 k r]); 
        end
    end

    % after building all cores, return the TT
    transition_tt = cell2core(tt_tensor, core_cells);
    
    % index out xA and xB
    % in MATLAB, needs to create an array indexer programmatically
    arr_indexer = cell([1, 2*(N+2)]); 
    for i = 1:length(arr_indexer)
        arr_indexer{i} = ':'; % used for A(:,:)
    end
    
    % ordering: xN_2, xN_1, xN-1_2, xN-1_1, ..., x1_2, x1_1, x0_2, x0_1
    arr_indexer{2*1-1} = xB_ypos; 
    %arr_indexer{2*1-1} = k-xB_ypos+1; 
    arr_indexer{2*1}   = xB_xpos;
    %arr_indexer{2*1}   = k-xB_xpos+1;
    
    arr_indexer{2*(N+2)-1} = xA_ypos;
    %arr_indexer{2*(N+2)-1} = k-xA_ypos+1;
    arr_indexer{2*(N+2)}   = xA_xpos;
    %arr_indexer{2*(N+2)}   = k-xA_xpos+1;
    
    transition_tt = transition_tt(arr_indexer{:});
    assert( length(transition_tt.n) == 2*N, ...
        "indexing failed, needs to rid of 2 dimensions" );
end


function B = drift(x, c)
    % nonequilibrium drift used in the paper
    %
    % Input:
    %       x,              (N x 2) denoting the 2d points
    %       A,              (4 x 1) scalar parameters used
    %       mu,             (4 x 2) denoting the four parameters used
    %       c,              (scalar) nonconservative coefficient
    %
    %
    % Output:
    %       B,              (N x 2) denoting the nonequilibrium dynamics
    mu1 = [0,1/3];
    mu2 = [0,5/3]; 
    mu3 = [-1,0];  
    mu4 = [1,0];
    %A1 = 30; A2 = -30; A3 = 50; A4 = 50; 
    A1 = 30; A2 = -30; A3 = -50; A4 = -50; A5 = 0.2;
    
    % drift part
    f = c * x(:, [2,1]);
    f(:,1) = -f(:,1);
    
    % get gradients from sum of Gaussian functions
    [grad1_dx, grad1_dy, ~] = gaussian_function(x, A1, mu1);
    [grad2_dx, grad2_dy, ~] = gaussian_function(x, A2, mu2);
    [grad3_dx, grad3_dy, ~] = gaussian_function(x, A3, mu3);
    [grad4_dx, grad4_dy, ~] = gaussian_function(x, A4, mu4);
    % gradient part 1
    grad1 = [grad1_dx, grad1_dy];
    grad2 = [grad2_dx, grad2_dy];
    grad3 = [grad3_dx, grad3_dy];
    grad4 = [grad4_dx, grad4_dy];
    
    % four Gaussians + additional penalty on mu1
    B = - (grad1 + grad2 + grad3 + grad4 + 4*A5*(x - mu1).^3) + f;
end