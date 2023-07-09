import numpy as np
import scipy as sp
import scipy.io
import scipy.stats
# pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

# very high precision
torch.set_default_dtype(torch.float64)
# use multi-threads
torch.set_num_threads(10)

# system
import os
import gc
import math
# supress warnings
import warnings
warnings.filterwarnings("ignore")

# define Double Rosenbrock function for density evaluation
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

def rosen(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. """
    d = np.shape(theta)[1]
    # formula from paper: An n-dimensional Rosenbrock Distribution
    # unknown partition function
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
    return torch.exp(-0.5*result)

def rosen_minus(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. 
    The upper half of double Rosenbrock. """
    d = np.shape(theta)[1]
    # formula from paper: An n-dimensional Rosenbrock Distribution
    # unknown partition function
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
    return torch.exp(-0.5*result)

def double_rosen(theta):
    """ theta is a torch tensor in R^(Nxd). Assuming from [-1,1]. 
    Mixture Rosenbrock distribution: 0.5 * rosen+(theta) + 0.5 * rosen-(theta) """
    return 0.5 * rosen(theta) + 0.5 * rosen_minus(theta)

def log_double_rosen(theta):
    E_plus = -0.5 * rosen_energy(theta)
    E_minus = -0.5 * rosen_minus_energy(theta)
    c = torch.max(E_plus, E_minus)
    log_targ_density = c + torch.log(torch.exp(E_plus - c) + torch.exp(E_minus - c)) + np.log(0.5)
    return log_targ_density


if __name__ == '__main__':
    # load data
    X = torch.Tensor(scipy.io.loadmat('./double_rosen_data.mat')['X'].T)
    print(">>> Python Data Loaded, of size: {}".format(X.shape))
    # evaluate
    p = double_rosen(X)
    log_p = log_double_rosen(X)
    print(" Evaluated: \n")
    print("  Target Distribution  |  Log Target Distribution  \n\n")
    save_p, save_log_p = [], []
    for it1, it2 in zip(p, log_p):
        save_p.append(it1.item())
        save_log_p.append(it2.item())
        print(np.log(it1.item()), it2.item())
    # save results
    print("Error (original - exp(log-sum-exp)) = {}".format(torch.norm(torch.Tensor(save_p) -\
         torch.exp(torch.Tensor(save_log_p)))))
    scipy.io.savemat('./double_rosen_python_result.mat', {'p_python': save_p, 'log_p_python': save_log_p})
    
