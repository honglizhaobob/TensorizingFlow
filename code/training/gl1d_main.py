"""
Date: 12/31/2021

    Main driver code to run examples for Tensorizing Flow (Ginzburg-Landau 1D).

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
    all_seeds = np.arange(1, 3)
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("=========== Running seed = {}".format(seed))
        if experiment == 'planar':
            ##################################
            # GL 1d experiment with PlanarFlow
            ##################################
            gl1d_tt_rank = 3
            # load dataset (Tensorizing)
            ginz1d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank))
            # initialize NF model
            ginz1d_tf = NormalizingFlow(dim=ginz1d_tf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=100)
            # begin training (Tensorizing)
            ginz1d_tf_planar_report = train(ginz1d_tf_dataset, ginz1d_tf, ginzburg_landau1d_logpdf, 
                num_epochs=300,
                batch_size=2**9,
                verbose=True,
                lr=5e-3, 
                use_scheduler=False,
                grad_clip=1e+4)
            # load dataset (Normalizing)
            ginz1d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank), \
                gaussian_data=True)
            ginz1d_nf = NormalizingFlow(dim=ginz1d_nf_dataset.dim, blocks=[PlanarFlow], \
                                    flow_length=100)
            # begin training (Normalizing)
            ginz1d_nf_planar_report = train(ginz1d_nf_dataset, ginz1d_nf, ginzburg_landau1d_logpdf, 
                num_epochs=300,
                batch_size=2**9,
                verbose=True,
                lr=5e-3, 
                use_scheduler=False,
                grad_clip=1e+4)
            scipy.io.savemat("./report/seed{}_ginz1d_rk{}_tf_planar_report.mat".format(seed, \
                gl1d_tt_rank), ginz1d_tf_planar_report)
            scipy.io.savemat("./report/seed{}_ginz1d_nf_planar_report.mat".format(seed), ginz1d_nf_planar_report)
            
        if experiment == 'realnvp':
            ##################################
            # GL 1d experiment with RealNVP
            ##################################
            gl1d_tt_rank = 3
            # load dataset (Tensorizing)
            ginz1d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank))
            # initialize NF model
            ginz1d_tf = NormalizingFlow(dim=ginz1d_tf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=3)
            # begin training (Tensorizing)
            ginz1d_tf_realnvp_report = train(ginz1d_tf_dataset, ginz1d_tf, ginzburg_landau1d_logpdf, 
                num_epochs=200,
                batch_size=950,
                verbose=True,
                lr=6e-4, 
                use_scheduler=True,
                grad_clip=1e+5)
            # load dataset (Normalizing)
            ginz1d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank), \
                gaussian_data=True)
            ginz1d_nf = NormalizingFlow(dim=ginz1d_nf_dataset.dim, blocks=REAL_NVP_BLOCKS, \
                                    flow_length=3)
            # begin training (Normalizing)
            ginz1d_nf_realnvp_report = train(ginz1d_nf_dataset, ginz1d_nf, ginzburg_landau1d_logpdf, 
                num_epochs=200,
                batch_size=950,
                verbose=True,
                lr=6e-4, 
                use_scheduler=True,
                grad_clip=1e+5)
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_ginz1d_rk{}_tf_realnvp_report.mat".format(seed, \
                gl1d_tt_rank), ginz1d_tf_realnvp_report)
            scipy.io.savemat("./report/seed{}_ginz1d_nf_realnvp_report.mat".format(seed), ginz1d_nf_realnvp_report)

        if experiment == 'resnet':
            ##################################
            # GL 1d experiment with ResNet
            ##################################
            gl1d_tt_rank = 3

            ######### hyperparameters (tune these) #########
            #########                              #########
            #########                              #########
            num_epochs = 500
            batch_size = 2**8
            lr = 5e-3
            flow_length = 3
            ##########                             #########
            #########                              #########
            #########                              #########

            # load dataset (Tensorizing)
            ginz1d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank))
            # initialize NF model
            ginz1d_tf = NormalizingFlow(dim=ginz1d_tf_dataset.dim, blocks=RESNET_BLOCKS2_GL1D, \
                                    flow_length=flow_length)
            # begin training (Tensorizing)
            ginz1d_tf_realnvp_report = train(ginz1d_tf_dataset, ginz1d_tf, ginzburg_landau1d_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr, 
                use_scheduler=True)

            scipy.io.savemat("./report/seed{}_ginz1d_rk{}_tf_resnet_report.mat".format(seed, \
                gl1d_tt_rank), ginz1d_tf_realnvp_report)

            # load dataset (Normalizing)
            ginz1d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_samples_rk{}.mat".format(gl1d_tt_rank), \
                gaussian_data=True)
            ginz1d_nf = NormalizingFlow(dim=ginz1d_nf_dataset.dim, blocks=RESNET_BLOCKS2_GL1D, \
                                    flow_length=flow_length)
            # begin training (Normalizing)
            ginz1d_nf_realnvp_report = train(ginz1d_nf_dataset, ginz1d_nf, ginzburg_landau1d_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr, 
                use_scheduler=True)
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_ginz1d_nf_resnet_report.mat".format(seed), ginz1d_nf_realnvp_report)


        


