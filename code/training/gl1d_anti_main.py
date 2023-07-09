"""
Date: 03/30/2021

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
    all_seeds = np.arange(9, 15)
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("=========== Running seed = {}".format(seed))
        if experiment == 'resnet':
            ##################################
            # GL 1d experiment with ResNet
            ##################################

            ######### hyperparameters (tune these) #########
            #########                              #########
            #########                              #########
            num_epochs = 300
            batch_size = 2**7
            lr = 8e-5
            flow_length = 4
            ##########                             #########
            #########                              #########
            #########                              #########




            # load dataset (Tensorizing)
            ginz1d_tf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_anti_truncated.mat")
            # initialize NF model
            ginz1d_tf = NormalizingFlow(dim=ginz1d_tf_dataset.dim, blocks=RESNET_BLOCKS2_GL1D_ANTI, \
                                    flow_length=flow_length)
            # begin training (Tensorizing)
            ginz1d_tf_resnet_report = train(ginz1d_tf_dataset, ginz1d_tf, ginzburg_landau1d_anti_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr, 
                use_scheduler=True,
                grad_clip=1e+4)

            scipy.io.savemat("./report/seed{}_ginz1d_anti_tf_resnet_report.mat".format(seed), ginz1d_tf_resnet_report)

            # load dataset (Normalizing)
            ginz1d_nf_dataset = utils.datasets.TensorizingFlowDataset("gl1d_anti_truncated.mat", \
                gaussian_data=True)
            ginz1d_nf = NormalizingFlow(dim=ginz1d_nf_dataset.dim, blocks=RESNET_BLOCKS2_GL1D_ANTI, \
                                    flow_length=flow_length)
            # begin training (Normalizing)
            ginz1d_nf_resnet_report = train(ginz1d_nf_dataset, ginz1d_nf, ginzburg_landau1d_anti_logpdf, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=True,
                lr=lr, 
                use_scheduler=True,
                grad_clip=1e+4)
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_ginz1d_anti_nf_resnet_report.mat".format(seed), ginz1d_nf_resnet_report)