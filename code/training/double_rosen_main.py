"""
Date: 12/31/2021

    Main driver code to run examples for Tensorizing Flow (Double Rosenbrock).

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
    
    # which exp to run
    experiment = 'planar'

    # run across different seeds
    # fix seed
    all_seeds = np.arange(1, 10)
    all_seeds = [1] # for quick experiment
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("=========== Running seed = {}".format(seed))
        if experiment == 'planar':
            ###############################################
            # Double rosenbrock experiment with PlanarFlow
            ###############################################
            dr_tt_rank = 4
            # load dataset (Tensorizing)
            dr_tf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank))
            # initialize NF model
            dr_tf = NormalizingFlow(dim=dr_tf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=128)
            # begin training (Tensorizing)
            dr_tf_planar_report = train(dr_tf_dataset, dr_tf, double_rosen_logpdf, 
                num_epochs=1000,
                batch_size=2**8,
                verbose=True,
                lr=1e-4, 
                use_scheduler=False,
                grad_clip=1e+4)
            # load dataset (Normalizing)
            dr_nf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank), \
                gaussian_data=True)
            dr_nf = NormalizingFlow(dim=dr_nf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=128)
            # begin training (Normalizing)
            dr_nf_planar_report = train(dr_nf_dataset, dr_nf, double_rosen_logpdf, 
                num_epochs=1000,
                batch_size=2**8,
                verbose=True,
                lr=1e-4, 
                use_scheduler=False,
                grad_clip=1e+4)
            scipy.io.savemat("./report/seed{}_double_rosen_rk{}_tf_planar_report.mat".format(seed, \
                dr_tt_rank), dr_tf_planar_report)
            scipy.io.savemat("./report/seed{}_double_rosen_nf_planar_report.mat".format(seed), dr_nf_planar_report)
            
        if experiment == 'realnvp':
            ############################################
            # Double rosenbrock experiment with RealNVP
            ############################################
            dr_tt_rank = 4
            # load dataset (Tensorizing)
            dr_tf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank))
            # initialize NF model
            dr_tf = NormalizingFlow(dim=dr_tf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=1)
            # begin training (Tensorizing)
            dr_tf_realnvp_report = train(dr_tf_dataset, dr_tf, double_rosen_logpdf, 
                num_epochs=150,
                batch_size=2**9,
                verbose=True,
                lr=1e-3, 
                use_scheduler=True,
                grad_clip=1e+3)
            # load dataset (Normalizing)
            dr_nf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank), \
                gaussian_data=True)
            dr_nf = NormalizingFlow(dim=dr_nf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=1)
            # begin training (Normalizing)
            dr_nf_realnvp_report = train(dr_nf_dataset, dr_nf, double_rosen_logpdf, 
                num_epochs=150,
                batch_size=2**9,
                verbose=True,
                lr=1e-3, 
                use_scheduler=True,
                grad_clip=1e+3)
            
            # save Double rosenbrock report
            scipy.io.savemat("./report/seed{}_double_rosen_rk{}_tf_realnvp_report.mat".format(seed, \
                dr_tt_rank), dr_tf_realnvp_report)
            scipy.io.savemat("./report/seed{}_double_rosen_nf_realnvp_report.mat".format(seed), dr_nf_realnvp_report)

        if experiment == 'resnet':
            ############################################
            # Double rosenbrock experiment with ResNet
            ############################################
            dr_tt_rank = 4
            # load dataset (Tensorizing)
            dr_tf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank))
            # initialize NF model
            dr_tf = NormalizingFlow(dim=dr_tf_dataset.dim, blocks=RESNET_BLOCKS2_DROSEN, \
                                    flow_length=4)
            # begin training (Tensorizing)
            dr_tf_realnvp_report = train(dr_tf_dataset, dr_tf, double_rosen_logpdf, 
                num_epochs=1000,
                batch_size=2**9,
                verbose=True,
                lr=5e-5, 
                use_scheduler=True,
                grad_clip=1e+4)

            # save Double rosenbrock report
            scipy.io.savemat("./report/seed{}_double_rosen_rk{}_tf_resnet_report.mat".format(seed, \
                dr_tt_rank), dr_tf_realnvp_report)


            # load dataset (Normalizing)
            dr_nf_dataset = utils.datasets.TensorizingFlowDataset("double_rosen_samples_rk{}.mat".format(dr_tt_rank), \
                gaussian_data=True)
            dr_nf = NormalizingFlow(dim=dr_nf_dataset.dim, blocks=RESNET_BLOCKS2_DROSEN, \
                                    flow_length=4)
            # begin training (Normalizing)
            dr_nf_realnvp_report = train(dr_nf_dataset, dr_nf, double_rosen_logpdf, 
                num_epochs=1000,
                batch_size=2**9,
                verbose=True,
                lr=5e-5, 
                use_scheduler=True,
                grad_clip=1e+3)
            
            # save Double rosenbrock report
            scipy.io.savemat("./report/seed{}_double_rosen_nf_resnet_report.mat".format(seed), dr_nf_realnvp_report)
