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


def log_rosen(theta):
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
    return -0.5 * result 

if __name__ == '__main__':
    # load data
    X = torch.Tensor(scipy.io.loadmat('./rosen_data.mat')['X'].T)
    print(">>> Python Data Loaded, of size: {}".format(X.shape))
    # evaluate
    p = rosen(X)
    log_p = log_rosen(X)
    print(" Evaluated: \n")
    print("  Target Distribution  |  Log Target Distribution  \n\n")
    save_p, save_log_p = [], []
    for train_loss, test_loss in zip(p, log_p):
        save_p.append(train_loss.item())
        save_log_p.append(test_loss.item())
        print(train_loss.item(), test_loss.item())
    # save results
    scipy.io.savemat('./rosen_python_result.mat', {'p_python': save_p, 'log_p_python': save_log_p})
    
