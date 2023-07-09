"""
Date: 12/31/2021

    Main driver code to run examples for Tensorizing Flow (Ginzburg-Landau 2D).

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
torch.manual_seed(0)

# import plotting if available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("utils::main.py: Matplotlib is not available on this machine. ")

if __name__ == "__main__":
    # main driver code

    # make sure ./report folder exists
    assert os.path.isdir("./report/"), "Please make sure './report' folder is created to store report data. "
    
    # which experiment to run
    experiment = 'resnet'

    # run across different seeds
    # fix seed
    all_seeds = np.arange(1, 2)
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("=========== Running seed = {}".format(seed))
            
        if experiment == 'realnvp':
            ##################################
            # GL 2d experiment with RealNVP
            ##################################
            gl2d_tt_rank = 3
            # load dataset (Tensorizing)
            ginz2d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl2d_samples_rk{}.mat".format(gl2d_tt_rank))
            # initialize NF model
            ginz2d_tf = NormalizingFlow(dim=ginz2d_tf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=3)
            # begin training (Tensorizing)
            ginz2d_tf_realnvp_report = train(ginz2d_tf_dataset, ginz2d_tf, ginzburg_landau2d_logpdf, 
                num_epochs=300,
                batch_size=2**9,
                verbose=True,
                lr=5e-4, 
                use_scheduler=True,
                grad_clip=1e+4)
            # load dataset (Normalizing)
            ginz2d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl2d_samples_rk{}.mat".format(gl2d_tt_rank), \
                gaussian_data=True)
            ginz2d_nf = NormalizingFlow(dim=ginz2d_nf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=3)
            # begin training (Normalizing)
            ginz2d_nf_realnvp_report = train(ginz2d_nf_dataset, ginz2d_nf, ginzburg_landau2d_logpdf, 
                num_epochs=300,
                batch_size=2**9,
                verbose=True,
                lr=5e-4, 
                use_scheduler=True,
                grad_clip=1e+4)
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_ginz2d_rk{}_tf_realnvp_report.mat".format(seed, \
                gl2d_tt_rank), ginz2d_tf_realnvp_report)
            scipy.io.savemat("./report/seed{}_ginz2d_nf_realnvp_report.mat".format(seed), ginz2d_nf_realnvp_report)

        if experiment == 'resnet':
            ##################################
            # GL 2d experiment with ResNet
            ##################################
            gl2d_tt_rank = 3

            ######### hyperparameters
            num_epochs = 500
            batch_size = 2**8
            lr = 4e-4
            flow_length = 6
            ##########


            # load dataset (Tensorizing)
            ginz2d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl2d_samples_rk{}.mat".format(gl2d_tt_rank))
            # initialize NF model
            ginz2d_tf = NormalizingFlow(dim=ginz2d_tf_dataset.dim, blocks=RESNET_BLOCKS2_GL2D, \
                                    flow_length=flow_length)
            # begin training (Tensorizing)
            ginz2d_tf_realnvp_report = train(ginz2d_tf_dataset, ginz2d_tf, ginzburg_landau2d_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr, 
                use_scheduler=True,
                grad_clip=1e+4)

            scipy.io.savemat("./report/seed{}_ginz2d_rk{}_tf_resnet_report.mat".format(seed, \
                gl2d_tt_rank), ginz2d_tf_realnvp_report)

            # load dataset (Normalizing)
            ginz2d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl2d_samples_rk{}.mat".format(gl2d_tt_rank), \
                gaussian_data=True)
            ginz2d_nf = NormalizingFlow(dim=ginz2d_nf_dataset.dim, blocks=RESNET_BLOCKS2_GL2D, \
                                    flow_length=flow_length)
            # begin training (Normalizing)
            ginz2d_nf_realnvp_report = train(ginz2d_nf_dataset, ginz2d_nf, ginzburg_landau2d_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr,
                use_scheduler=True,
                grad_clip=1e+4)
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_ginz2d_nf_resnet_report.mat".format(seed), ginz2d_nf_realnvp_report)