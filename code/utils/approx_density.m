function approx_data = approx_density(coeff_tt,x_grid)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author(s): Hongli Zhao
% 
% Given a tensor train coefficients assumed to have Legendre basis,
% contracts all coefficients with Legendre basis evaluated at 
% grid points (specified by `x_grid`) and returns a new tensor train.
% This routine effectively fix the continuous tensor train at points
% specified by `x_grid`.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ndim = coeff_tt.d;
    coeff_cores = core(coeff_tt);
    approx_data = cell(ndim,1);
    for i = 1:ndim
        legendre_ord = coeff_tt.n(i)-1;
        % matrix size (n x legendre_ord)
        U = get_legendre(x_grid{i},legendre_ord,true); 
        if i == 1
            tmp = U*coeff_cores{i};
            tmp = reshape(tmp,[1 size(tmp)]);
            
        elseif i == ndim
            tmp = U*coeff_cores{i};
            tmp = tmp';
        else
            tmp = U*coeff_cores{i};
            tmp = permute(tmp,[2 1 3]);
        end
        approx_data{i} = tmp;
    end
    approx_data = cell2core(tt_tensor(),approx_data);
end