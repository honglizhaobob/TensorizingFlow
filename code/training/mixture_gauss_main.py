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
    all_seeds = np.arange(1, 11)
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
            num_epochs = 200
            batch_size = 2**7
            lr = 2e-4
            flow_length = 3
            ##########                             #########
            #########                              #########
            #########                              #########

            # load dataset (Tensorizing)
            mixture_gauss_tf_dataset = utils.datasets.TensorizingFlowDataset("mixture_gaussian_patterned_truncated.mat", \
                                                              gaussian_data=False)
            # input mean and std to wrapper
            mu_input = np.array([mixture_gauss_tf_dataset.raw_dataset['mu1'][0], \
                                mixture_gauss_tf_dataset.raw_dataset['mu2'][0], \
                                mixture_gauss_tf_dataset.raw_dataset['mu3'][0], \
                                mixture_gauss_tf_dataset.raw_dataset['mu4'][0], \
                                mixture_gauss_tf_dataset.raw_dataset['mu5'][0]])
            std_input = np.array([mixture_gauss_tf_dataset.raw_dataset['covmat1'], \
                                mixture_gauss_tf_dataset.raw_dataset['covmat2'], \
                                mixture_gauss_tf_dataset.raw_dataset['covmat3'], \
                                mixture_gauss_tf_dataset.raw_dataset['covmat4'], \
                                mixture_gauss_tf_dataset.raw_dataset['covmat5']])
            def mixture_gaussian_logpdf_wrapper(theta, mu_list=mu_input, sig_list=std_input):
                return mixture_gaussian_logpdf(theta, mu_list, sig_list, scaling=2)
            
            # initialize NF model
            mixture_flow = NormalizingFlow(dim=mixture_gauss_tf_dataset.dim, blocks=RESNET_BLOCKS2_MIXTURE_GAUSSIAN, \
                                        flow_length=flow_length)
            
            
            # begin training
            mixture_report = train(mixture_gauss_tf_dataset, mixture_flow, mixture_gaussian_logpdf_wrapper, 
                    num_epochs=num_epochs,
                    batch_size=2**7,
                    verbose=True,
                    lr=lr,
                    use_scheduler=True,
                    grad_clip=1e+4)

            scipy.io.savemat("./report/seed{}_mixture_gaussian_tf_resnet_report.mat".format(seed), mixture_report)


            # Normalizing flow dataset
            mixture_gauss_nf_dataset = utils.datasets.TensorizingFlowDataset("mixture_gaussian_patterned_truncated.mat", \
                                                              gaussian_data=True)
            # initialize NF model
            mixture_flow = NormalizingFlow(dim=mixture_gauss_nf_dataset.dim, blocks=RESNET_BLOCKS2_MIXTURE_GAUSSIAN, \
                                        flow_length=flow_length)
            # begin training
            mixture_report2 = train(mixture_gauss_nf_dataset, mixture_flow, mixture_gaussian_logpdf_wrapper, 
                    num_epochs=num_epochs,
                    batch_size=2**7,
                    verbose=True,
                    lr=lr, 
                    use_scheduler=True,
                    grad_clip=1e+4)
    
            
            # save GL 1d report
            scipy.io.savemat("./report/seed{}_mixture_gaussian_nf_resnet_report.mat".format(seed), mixture_report2)