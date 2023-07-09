""" Various target functions for energy evaluation (used in training loss).

Note: all energy functions are scaled in [-1, 1] for Legendre basis. Certain 
model parameters are pre-defined, check corresponding MATLAB routines.

# Refactored: 12/28/2021
"""
# import pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

# scipy
import scipy
import scipy.io

# numpy
import numpy as np

# very high precision
torch.set_default_dtype(torch.float64)
import torch.optim as optim

# set number of threads (8 threads should work on most computers)
torch.set_num_threads(8)

# supress warnings
import warnings
warnings.filterwarnings("ignore")

## Ginzburg-Landau 1D
def ginzburg_landau_energy1d(U, delta=0.04, scaling=2):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs energy as shape (N x 1) tensor. Parameter choices from Yian et al.
    """
    # make row vector
    U = torch.Tensor(U)
    d = U.shape[1]
    U = scaling * U

    # compute stepsize
    h = 1 / (d+1)
    # compute energy with U_0 and U_d+1
    V = ( (delta/2) * ( ((1 / h) * (U[:,1:d] - U[:,0:d-1]))**2 ) + \
    (1/(4 * delta)) * ( (1 - U[:,1:d]**2 )**2 ) ).sum(1)
    return V

def ginzburg_landau1d_logpdf(U, temp=8):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs Log likelihood as shape (N x 1) tensor, parameter choice from Yian et al.
    """
    # Boltzmann energy
    E = ginzburg_landau_energy1d(U)
    beta = 1 / temp
    return -beta * E

## Ginzburg-Landau 1D antiferromagnetic
def ginzburg_landau_anti_energy1d(U, delta=0.04, scaling=4.5):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs energy as shape (N x 1) tensor. Parameter choices from Yian et al.
    """
    # make row vector
    U = torch.Tensor(U)
    d = U.shape[1]
    U = scaling * U

    # compute stepsize
    h = 1 / (d+1)
    # append first and last dimension with 0
    U = torch.cat([torch.zeros([U.shape[0],1]), U], 1)
    U = torch.cat([U, torch.zeros([U.shape[0],1])], 1)      # now of size N x (d+2)
    # compute energy with U_0 and U_d+1
    V = ( (-delta/2) * ( ((1 / h) * (U[:,1:d+2] - U[:,0:d+1]))**2 ) + \
    (1/(4 * delta)) * ( (1 - U[:,1:d+2]**2 )**2 ) ).sum(1)
    return V

def ginzburg_landau1d_anti_logpdf(U, temp=16):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs Log likelihood as shape (N x 1) tensor, parameter choice from Yian et al.
    """
    # Boltzmann energy
    E = ginzburg_landau_anti_energy1d(U)
    beta = 1 / temp
    return -beta * E

## Ginzburg-Landau spin glass with random [-1,1] coefficients (Added: 03/20/2022)
def ginzburg_landau_spin1d(U, delta=0.04, scaling=3.5, \
    spin_coeffs=scipy.io.loadmat("./utils/spin_glass_coefs2.mat")['coefs'][0]):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs energy as shape (N x 1) tensor. Parameter choices from Yian et al.
    """
    # make row vector
    U = torch.Tensor(U)
    _coefs = torch.Tensor(spin_coeffs)
    # append [1, _coefs, 1]
    _coefs = torch.cat([torch.Tensor([1]), _coefs])
    _coefs = torch.cat([_coefs, torch.Tensor([1])])
    d = U.shape[1]
    U = scaling * U
    # append first and last dimension with 0
    U = torch.cat([torch.zeros([U.shape[0],1]), U], 1)
    U = torch.cat([U, torch.zeros([U.shape[0],1])], 1)      # now of size N x (d+2)

    # compute stepsize
    h = 1 / (d+1)
    # compute energy with U_0 and U_d+1
    V = ( (delta/2) * ( _coefs * ((1 / h) * (U[:,1:d+2] - U[:,0:d+1]))**2 ) + \
    (1/(4 * delta)) * ( (1 - U[:,1:d+2]**2 )**2 ) ).sum(1)
    return V

def ginzburg_landau1d_spin_logpdf(U, temp=16):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs Log likelihood as shape (N x 1) tensor, parameter choice from Yian et al.
    """
    # Boltzmann energy
    E = ginzburg_landau_spin1d(U)
    beta = 1 / temp
    return -beta * E



## Ginzburg-Landau 2D
def ginzburg_landau_energy2d(U, delta=0.04, scaling=2):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    
    Outputs energy as shape (N x 1) tensor
    
    """
    # make row vector
    U = torch.Tensor(U)
    U = scaling * U
    N = U.shape[0]
    d = U.shape[1]
    # dimension must be a perfect square
    board_dim = int(np.sqrt(d) + 0.5) # spatial dimension, instead of the PDF dimension
    assert board_dim**2 == d, "dimension is not a perfect square"
    # compute stepsize
    h = 1 / (board_dim+1)
    # torch reshape is different from MATLAB reshape
    # MATLAB is by column, torch is by row
    # we use MATLAB's format here, so need to transpose the result of torch.reshape
    U_2d = U.reshape([N, board_dim, board_dim])
    U_2d = torch.transpose(U_2d, 1, 2)
    # add boundary values
    U_2d_save = U_2d.clone()
    U_2d = torch.zeros([N, board_dim+2, board_dim+2]) # including U0, Ud+1
    U_2d[:,1:board_dim+1,1:board_dim+1] = U_2d_save
    # (u(x,y), x = 0, x = 1)
    U_2d[:, 0, :] = 1
    U_2d[:, -1, :] = 1
    # (u(x,y), y = 0, y = 1)
    U_2d[:, :, 0] = -1
    U_2d[:, :, -1] = -1
    # compute energy with U_0 and U_d+1
    u_x = (1 / h) * (U_2d[:, 1:board_dim+2, :] - U_2d[:, 0:board_dim+1, :])
    u_y = (1 / h) * (U_2d[:, :, 1:board_dim+2] - U_2d[:, :, 0:board_dim+1])
    V = (delta/2)*( (u_x ** 2).sum([1,2]) ) + \
     (delta/2) * ( (u_y**2).sum([1,2]) ) + \
    (1/(4*delta)) * ((1 - U_2d ** 2) ** 2).sum([1,2])
    return V

def ginzburg_landau2d_logpdf(U, temp=1.0):
    """ computes the GL energy (1d) for U, U is an torch.Tensor of shape N x d. 
    Outputs Log likelihood as shape (N x 1) tensor, parameter choice from Yian et al.

    (New Note 12/24/2022): samples have temperature 5, push it to temperature 1.
    """
    # Boltzmann energy
    E = ginzburg_landau_energy2d(U)
    beta = 1 / temp
    return -beta * E

## Rosenbrock
def rosen_energy(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. """
    d = np.shape(theta)[1]
    result = 0
    for k in range(d-1):
        if k <= d-4:
            # theta1, theta2, ..., theta_d-4, theta_d-3 are scaled by 2
            result += (2 * theta[:, k])**2 + ( (2 * theta[:, k+1]) + 5 * ( ( 2*theta[:, k] ) ** 2 + 1 ) )**2
        elif k == d-3:
            # theta d-2 is scaled by 2, theta d-1 is scaled by 7
            result += (2*theta[:, k])**2 + ( (7*theta[:, k+1]) + 5 * ( ( 2*theta[:, k] ) ** 2 + 1 ) )**2
        else:
            # theta d-1 is scaled by 7, theta d is scaled by 200
            result += (7*theta[:, k])**2 + ( (200*theta[:, k+1]) + 5 * ( ( 7*theta[:, k] ) ** 2 + 1 ) )**2
    return result

def rosen_logpdf(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. Returns
    log likelihood of Rosenbrock target. """
    return -0.5 * rosen_energy(theta)


## Double Rosenbrock (uses stable log-sum-exp evaluation)
def rosen_minus_energy(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. """
    d = np.shape(theta)[1]
    result = 0
    for k in range(d-1):
        if k <= d-4:
            # theta1, theta2, ..., theta_d-4, theta_d-3 are scaled by 2
            result += (2 * theta[:, k])**2 + ( (2 * theta[:, k+1]) + 5 * ( ( 2*theta[:, k] ) ** 2 + 1 ) )**2
        elif k == d-3:
            # theta d-2 is scaled by 2, theta d-1 is scaled by 7
            result += (2*theta[:, k])**2 + ( (7*theta[:, k+1]) + 5 * ( ( 2*theta[:, k] ) ** 2 + 1 ) )**2
        else:
            # theta d-1 is scaled by 7, theta d is scaled by 200
            result += (7*theta[:, k])**2 + ( (200*(-theta[:, k+1])) + 5 * ( ( 7*theta[:, k] ) ** 2 + 1 ) )**2
    return result

def double_rosen_logpdf(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. Returns 
    log likelihood of mixture Rosenbrock target. """
    E_plus = -0.5 * rosen_energy(theta)
    E_minus = -0.5 * rosen_minus_energy(theta)
    c = torch.max(E_plus, E_minus)
    return c + torch.log(torch.exp(E_plus - c) + torch.exp(E_minus - c)) + np.log(0.5)

## Additional Examples 1: 5-modal mixture Gaussian
def mixture_gaussian_logpdf(theta, mu, covmat, scaling=2):
    """ evaluates log PDF (unnormalized) at theta, given all means
    and covariances of each mixture Gaussian component. Uses LSE 
    evaluation trick. 

    theta is R^(N x d)
    
    """
    # mu and covmat should be lists containing mean and covariance for 
    # each Gaussian in the mixture
    U = scaling * torch.Tensor(theta)
    # number of dimensions
    N = np.shape(U)[0]
    d = np.shape(U)[1]
    num_gaussians = len(mu)
    # use log-sum-exp trick, the form is:
    #
    #      c_1 * exp( (theta.T - mean1)^T @ sig1^-1 @ (theta.T - mean1) ) + ...
    #      c_M * exp( (theta.T - meanM)^T @ sigM^-1 @ (theta.T - meanM) )
    # need to find max theta.T@sig@theta value over M
    to_max = torch.zeros([N, num_gaussians])
    for i in range(num_gaussians):
        mu_i = torch.Tensor(mu[i])
        covmat_i = torch.Tensor(covmat[i])
        # normalizing constant for a single Gaussian
        const_i = torch.log(1/torch.sqrt(((2*torch.pi)**d) * torch.abs(torch.det(covmat_i))))
        # PDF for a single Gaussian
        to_max[:, i] = const_i - 0.5 * torch.diag((U-mu_i) @ torch.linalg.inv(covmat_i) @ (U-mu_i).T)
    # compute max over M (for each i = 1,2,...,N)
    x_max = torch.max(to_max, axis=1).values
    # subtract x_max from each computed exponents, then exponentiate, sum, log, and add back x_max
    return x_max + torch.log(torch.sum(torch.exp(to_max - x_max.reshape(-1, 1)), axis=1))

## Additional Examples: Double Rosenbrock 3D (uses stable log-sum-exp evaluation)
def double_rosen_logpdf3d(theta):
    """ theta is a torch tensor in R^(Nx3). Assuming from [-1,1]. Returns 
    log likelihood of mixture Rosenbrock target. """
    E_plus = -0.5 * ( 
        (20 * theta[:, 0]**2 + 7 * theta[:, 1] + 5)**2 + 
        (245 * theta[:, 1]**2 - 200 * theta[:, 2] + 5)**2 + 
        4 * theta[:, 0]**2 + 
        (49 * theta[:, 1]**2)
     )
    E_minus = -0.5 * (
        (20 * theta[:, 0]**2 + 7 * theta[:, 1] + 5)**2 + 
        (245 * theta[:, 1]**2 + 200 * theta[:, 2] + 5)**2 + 
        4 * theta[:, 0]**2 + 
        (49 * theta[:, 1]**2)
    )
    c = torch.max(E_plus, E_minus)
    return c + torch.log(torch.exp(E_plus - c) + torch.exp(E_minus - c)) + np.log(0.5)


## Non-equilibrium Path Sampling
def tpt_logpdf(U, beta=0.25, dt=0.05):
    """
        Wrapper, returns logpdf of the transition path Boltzmann-PDF
    """
    U = torch.tensor(U, requires_grad=True)
    R = 4 # scaling 
    return -path_action(R*U, beta, dt)

def path_action(U, beta=0.25, dt=0.05):
    """ computes the effective energy (path integral) for path sample U. 
    
    Inputs:
        U,                          (torch.Tensor) (M x 2*N) path samples, each row is a path
                                    with time discretization level N
        beta,                       (scalar)       inverse temperature, parameter
        dt,                         (scalar)       time discretization level, parameter
        
    Outputs:
        S,                          (torch.Tensor) (M x 1)   energy for each path of U
    """
    U = torch.Tensor(U) # make sure of formatting
    assert U.shape[1] % 2 == 0, "Dimension of path must be divisible by 2. "
    assert U.requires_grad, "U must require grad, for auto-differentiation. "
    xA = torch.tensor([[-1., 0.]]); xB = torch.tensor([[1., 0.]])
    M = int(U.shape[0])
    N = int(U.shape[1] // 2)
    # reshape to (M x N x 2), torch reshapes by row, hence the transpose
    U_2d = U.reshape([M, 2, N])
    U_2d = U_2d.transpose(1, 2) # U has size (M x N x 2)
    U_2d = U_2d.view(U_2d.shape[0], -1, 2)
    # get gradient of potential, has size (M x N x 2) the same as U_2d
    grad_V = torch.autograd.grad(V(U_2d).sum(), U_2d, create_graph=True)[0]
    # compute path integral with differencing
    S = torch.sum( ( (U_2d[:,1:,:] - U_2d[:,:-1,:]) / dt + grad_V[:,:-1,:] )**2, dim=(1, 2))
    # penalize boundary xA
    S += torch.sum(
            ( (U_2d[:,0,:] - xA) / dt + grad_V[:,0,:] )**2, dim=1)
    # penalize boundary xB
    S += torch.sum(
            ( (xB - U_2d[:,-1,:]) / dt + grad_V[:,-1,:] )**2, dim=1)
    return beta * dt * S


def brownian_bridge_path_action(U, beta=0.25, dt=0.05):
    """ computes the effective energy (path integral) for path sample U. 
    
    Inputs:
        U,                          (torch.Tensor) (M x 2*N) path samples, each row is a path
                                    with time discretization level N
        beta,                       (scalar)       inverse temperature, parameter
        dt,                         (scalar)       time discretization level, parameter
        
    Outputs:
        S,                          (torch.Tensor) (M x 1)   energy for each path of U
    """
    U = torch.Tensor(U) # make sure of formatting
    assert U.shape[1] % 2 == 0, "Dimension of path must be divisible by 2. "
    assert U.requires_grad, "U must require grad, for auto-differentiation. "
    xA = torch.tensor([[-1., 0.]]); xB = torch.tensor([[1., 0.]])
    M = int(U.shape[0])
    N = int(U.shape[1] // 2)
    # reshape to (M x N x 2), torch reshapes by row, hence the transpose
    U_2d = U.reshape([M, 2, N])
    U_2d = U_2d.transpose(1, 2) # U has size (M x N x 2)
    U_2d = U_2d.view(U_2d.shape[0], -1, 2)

    # compute path integral with differencing
    S = torch.sum( ( (U_2d[:,1:,:] - U_2d[:,:-1,:]) / dt )**2, dim=(1, 2))
    # penalize boundary xA
    S += torch.sum(
            ( (U_2d[:,0,:] - xA) / dt )**2, dim=1)
    # penalize boundary xB
    S += torch.sum(
            ( (xB - U_2d[:,-1,:]) / dt )**2, dim=1)
    return beta * dt * S

def drift(U):
    """ Nonequilibrium drift, currently none. """
    pass

def V(U):
    """ Three-hole potential (weighted sum of Gaussian) 
    
    Inputs:
        U,                          (torch.Tensor) (M x N x 2) path samples
        
    Outputs:
        V,                          (M x N) scalar, three-hole potential
    """
    # parameters for the Gaussian, make sure they match with MATLAB
    mu1 = torch.tensor([[0., 1./3.]])
    mu2 = torch.tensor([[0., 5./3.]])
    mu3 = torch.tensor([[-1., 0.]])
    mu4 = torch.tensor([[1., 0.]])
    A1 = 40.; A2 = -55.; A3 = -50.; A4 = -50.; A5 = 0.2
    # mixture Gaussian with penalty
    return g(U, mu1, A1) + g(U, mu2, A2) +\
        g(U, mu3, A3) + g(U, mu4, A4) + 0.2 * torch.sum((U - mu1)**4, dim=-1)
    
    
def g(U, mu, amp):
    """ Gaussian function. 
    
    Inputs:
        U,                          (torch.Tensor) (M x N x 2) A path in R^2
        mu,                         (torch.Tensor) (1 x 2) mean of Gaussian
        amp,                        (scalar) amplitude of Gaussian
        
    Outputs:
        g_U,                        (M x N) result of Gaussian function evaluated at U 
    """
    M = U.shape[0]; N = U.shape[1]
    return amp * torch.exp(-torch.sum((U - mu)**2, dim=-1))