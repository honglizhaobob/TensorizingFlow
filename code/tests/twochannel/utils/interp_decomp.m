function [Ax1, Bx2, r] = interp_decomp(K, tol)
    % Helper function, given kernel K(x1,x2) of size (m x n)
    % performs truncated SVD (can equivalently perform CUR)
    % as decoupling of x1, x2. Error from rank truncation is 
    % determined from TOL, and reported.
    %
    %  Inputs:
    %           K,                  (2d array) Kernel of size 
    %                               m x n describing the interaction
    %                               between two variables x1, x2
    %           tol,                (scalar) relative tolerance
    %                               for rank-truncation.
    %
    %  Outputs:
    %           Ax1, Bx2            (2d array) K(x1,x2) \approx
    %                               Ax1 * Bx2'
    %           r,                  (scalar)   number of ranks
    %                               preserved. Namely
    %                               Ax1 has size (m x r)
    %                               Bx2 has size (m x r)
    [u,s,v] = svd(K);
    Ax1 = u*s; 
    Bx2 = v;
    % determine most representative columns
    idx = find(diag(s)./max(diag(s)) > tol);
    r = length(idx);
    Ax1 = Ax1(:,idx);
    Bx2 = Bx2(:,idx);
    % report error from truncated SVD
    disp(strcat("> ========== Error from interpolative SVD = ", ...
        num2str(norm(Ax1*Bx2' - K))))
end