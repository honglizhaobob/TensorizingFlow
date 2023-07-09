"""
Date: 12/31/2021

    Main driver code to run examples for Tensorizing Flow (Rosenbrock).

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

# import plotting if available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("utils::main.py: Matplotlib is not available on this machine. ")

if __name__ == "__main__":
    # main driver code

    # which exp to run
    experiment = 'planar'

    # make sure ./report folder exists
    assert os.path.isdir("./report/"), "Please make sure './report' folder is created to store report data. "

    # run across different seeds
    # fix seed
    all_seeds = np.arange(1, 10)
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("=========== Running seed = {}".format(seed))
        if experiment == 'planar':
            #######################################
            # Rosenbrock experiment with PlanarFlow
            #######################################
            rosen_tt_rank = 5
            # load dataset (Tensorizing)
            rosen_tf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank))
            # initialize NF model
            rosen_tf = NormalizingFlow(dim=rosen_tf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=64)
            # begin training (Tensorizing)
            rosen_tf_planar_report = train(rosen_tf_dataset, rosen_tf, rosen_logpdf, 
                num_epochs=300,
                batch_size=2**8,
                verbose=True,
                lr=3e-3, 
                use_scheduler=False,
                grad_clip=1e+4)
            # load dataset (Normalizing)
            rosen_nf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank), \
                gaussian_data=True)
            rosen_nf = NormalizingFlow(dim=rosen_nf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=64)
            # begin training (Normalizing)
            rosen_nf_planar_report = train(rosen_nf_dataset, rosen_nf, rosen_logpdf, 
                num_epochs=300,
                batch_size=2**8,
                verbose=True,
                lr=3e-3, 
                use_scheduler=False,
                grad_clip=1e+4)
            
            # save Rosenbrock report
            scipy.io.savemat("./report/seed{}_rosen_rk{}_tf_planar_report.mat".format(seed, \
                rosen_tt_rank), rosen_tf_planar_report)
            scipy.io.savemat("./report/seed{}_rosen_nf_planar_report.mat".format(seed), rosen_nf_planar_report)
        
        if experiment == 'realnvp':
            #####################################
            # Rosenbrock experiment with RealNVP
            #####################################
            rosen_tt_rank = 5
            # load dataset (Tensorizing)
            rosen_tf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank))
            # initialize NF model
            rosen_tf = NormalizingFlow(dim=rosen_tf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=1)
            # begin training (Tensorizing)
            rosen_tf_realnvp_report = train(rosen_tf_dataset, rosen_tf, rosen_logpdf, 
                num_epochs=300,
                batch_size=2**7,
                verbose=True,
                lr=2e-3, 
                use_scheduler=True,
                grad_clip=1e+3)
            # load dataset (Normalizing)
            rosen_nf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank), \
                gaussian_data=True)
            rosen_nf = NormalizingFlow(dim=rosen_nf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=1)
            # begin training (Normalizing)
            rosen_nf_realnvp_report = train(rosen_nf_dataset, rosen_nf, rosen_logpdf, 
                num_epochs=300,
                batch_size=2**7,
                verbose=True,
                lr=2e-3, 
                use_scheduler=True,
                grad_clip=1e+3)
            
            scipy.io.savemat("./report/seed{}_rosen_rk{}_tf_realnvp_report.mat".format(seed, \
                rosen_tt_rank), rosen_tf_realnvp_report)
            scipy.io.savemat("./report/seed{}_rosen_nf_realnvp_report.mat".format(seed), rosen_nf_realnvp_report)

        if experiment == 'resnet':
            #####################################
            # Rosenbrock experiment with ResNet
            #####################################
            rosen_tt_rank = 5
            # load dataset (Tensorizing)
            rosen_tf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank))
            # initialize NF model
            rosen_tf = NormalizingFlow(dim=rosen_tf_dataset.dim, blocks=RESNET_BLOCKS2_ROSEN, \
                                    flow_length=2)
            # begin training (Tensorizing)
            rosen_tf_realnvp_report = train(rosen_tf_dataset, rosen_tf, rosen_logpdf, 
                num_epochs=150,
                batch_size=2**8,
                verbose=True,
                lr=1e-4, 
                use_scheduler=True,
                grad_clip=1e+4)

            scipy.io.savemat("./report/seed{}_rosen_rk{}_tf_resnet_report.mat".format(seed, \
                rosen_tt_rank), rosen_tf_realnvp_report)


            # load dataset (Normalizing)
            rosen_nf_dataset = utils.datasets.TensorizingFlowDataset("rosen_samples_rk{}.mat".format(rosen_tt_rank), \
                gaussian_data=True)
            rosen_nf = NormalizingFlow(dim=rosen_nf_dataset.dim, blocks=RESNET_BLOCKS2_ROSEN, \
                                    flow_length=2)
            # begin training (Normalizing)
            rosen_nf_realnvp_report = train(rosen_nf_dataset, rosen_nf, rosen_logpdf, 
                num_epochs=150,
                batch_size=2**8,
                verbose=True,
                lr=1e-4, 
                use_scheduler=True,
                grad_clip=1e+4)
            
            scipy.io.savemat("./report/seed{}_rosen_nf_resnet_report.mat".format(seed), rosen_nf_realnvp_report)


