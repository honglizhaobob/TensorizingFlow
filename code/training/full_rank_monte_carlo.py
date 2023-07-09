"""
Date: 01/08/2021

    Main driver code to estimate statistics of untruncated TT via Monte-Carlo.

Author: Hongli Zhao, honglizhaobob@uchicago.edu
"""
# import utility functions
import utils
from utils.flow_models import *
from utils.target import *
from utils.training import *

# import pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform
# set number of threads
torch.set_num_threads(8)

# import numerical libraries
import scipy
import scipy.io
import numpy as np

# import I/O
import os
import sys
# set number of threads
torch.set_num_threads(8)

# prevent memory leak
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# fix seed
np.random.seed(8)
torch.manual_seed(8)

# import plotting if available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("utils::main.py: Matplotlib is not available on this machine. ")

if __name__ == "__main__":
    # main driver code

    # make sure ./data folder exists
    assert os.path.isdir("./data/"), "Please make sure './data' folder is created to store estimate data. "
    
    ## GL 1d stats
    def loss(x, prior_logpdf, targ_logpdf=ginzburg_landau1d_logpdf):
        """ evaluate initial KL divergence between posterior distribution (NF + prior) 
        and target. x is samples without flow. This is a Monte-Carlo estimation of the 
        log partition function. """
        return (prior_logpdf - targ_logpdf(x)).mean()
    gl1d_data = scipy.io.loadmat("./data/gl1d_fullrank_exact_gauss_samples.mat")
    gl1d_mean = gl1d_data['X'].mean(1)
    gl1d_std = gl1d_data['X'].std(1)
    # KL Div
    gl1d_data_torch = torch.Tensor(gl1d_data['X']).T
    gl1d_data_like_torch = torch.log(torch.Tensor(gl1d_data['likes']).reshape(1, -1)[0])
    gl1d_optim_loss = loss(gl1d_data_torch, gl1d_data_like_torch).detach().numpy().item()

    gl1d_stat = {'mean': gl1d_mean, 'std': gl1d_std, 'loss': gl1d_optim_loss}

    ## GL 2d stats
    def loss(x, prior_logpdf, targ_logpdf=ginzburg_landau2d_logpdf):
        """ evaluate initial KL divergence between posterior distribution (NF + prior) 
        and target. x is samples without flow. This is a Monte-Carlo estimation of the 
        log partition function. """
        return (prior_logpdf - targ_logpdf(x)).mean()
    gl2d_data = scipy.io.loadmat("./data/gl2d_fullrank_exact_gauss_samples.mat")
    gl2d_mean = gl2d_data['X'].mean(1)
    gl2d_std = gl2d_data['X'].std(1)
    # KL Div
    gl2d_data_torch = torch.Tensor(gl2d_data['X']).T
    gl2d_data_like_torch = torch.log(torch.Tensor(gl2d_data['likes']).reshape(1, -1)[0])
    gl2d_optim_loss = loss(gl2d_data_torch, gl2d_data_like_torch).detach().numpy().item()

    gl2d_stat = {'mean': gl2d_mean, 'std': gl2d_std, 'loss': gl2d_optim_loss}

    ## Rosen stats
    def loss(x, prior_logpdf, targ_logpdf=rosen_logpdf):
        """ evaluate initial KL divergence between posterior distribution (NF + prior) 
        and target. x is samples without flow. This is a Monte-Carlo estimation of the 
        log partition function. """
        return (prior_logpdf - targ_logpdf(x)).mean()
    rosen_data = scipy.io.loadmat("./data/rosen_fullrank_exact_gauss_samples.mat")
    rosen_mean = rosen_data['X'].mean(1)
    rosen_std = rosen_data['X'].std(1)
    # KL Div
    rosen_data_torch = torch.Tensor(rosen_data['X']).T
    rosen_data_like_torch = torch.log(torch.Tensor(rosen_data['likes']).reshape(1, -1)[0])
    rosen_optim_loss = loss(rosen_data_torch, rosen_data_like_torch).detach().numpy().item()

    rosen_stat = {'mean': rosen_mean, 'std': rosen_std, 'loss': rosen_optim_loss}

    ## Double Rosen stats
    def loss(x, prior_logpdf, targ_logpdf=double_rosen_logpdf):
        """ evaluate initial KL divergence between posterior distribution (NF + prior) 
        and target. x is samples without flow. This is a Monte-Carlo estimation of the 
        log partition function. """
        return (prior_logpdf - targ_logpdf(x)).mean()
    drosen_data = scipy.io.loadmat("./data/double_rosen_fullrank_exact_gauss_samples.mat")
    drosen_mean = drosen_data['X'].mean(1)
    drosen_std = drosen_data['X'].std(1)
    # KL Div
    drosen_data_torch = torch.Tensor(drosen_data['X']).T
    drosen_data_like_torch = torch.log(torch.Tensor(drosen_data['likes']).reshape(1, -1)[0])
    drosen_optim_loss = loss(drosen_data_torch, drosen_data_like_torch).detach().numpy().item()

    drosen_stat = {'mean': drosen_mean, 'std': drosen_std, 'loss': drosen_optim_loss}

    # save data
    scipy.io.savemat("./data/full_rank_stats.mat", \
        {"gl1d": gl1d_stat, "gl2d": gl2d_stat, "rosen": rosen_stat, "double_rosen": drosen_stat})
    