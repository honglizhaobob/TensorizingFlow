# Main Python script to verify that target
# function evaluation matches with MATLAB's output
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

# numerical libs
import scipy
import scipy.io

# import I/O
import os
import sys

# set seed
seed = 2
np.random.seed(seed)
torch.manual_seed(0)

# load comparison data
compare_energy_brownian_bridge = scipy.io.loadmat("./test_brownian_energy.mat")['all_energy'].flatten()
compare_energy_transition_path = scipy.io.loadmat("./test_tpt_energy.mat")['all_energy'].flatten()

# compute energy in Python for samples
data = scipy.io.loadmat("./data/pathsampling_fullrank_small.mat")
R = data['R'][0][0]
sample_paths = R * data['X'].T    # rescale back to original domain

# evaluate energy
python_energy_brownian_bridge = brownian_bridge_path_action(sample_paths).detach().numpy().flatten()
python_energy_transition_path = path_action(sample_paths).detach().numpy().flatten()

if __name__ == '__main__':
    print(">>> Brownian bridge energy: norm| Python Eval - MATLAB Eval | = {}".format(np.linalg.norm(
        python_energy_brownian_bridge - compare_energy_brownian_bridge
    )))
    print()
    print(">>> Displaying values \n\n")
    print(python_energy_brownian_bridge)
    print(compare_energy_brownian_bridge)

    print(">>> Transition path energy: norm| Python Eval - MATLAB Eval | = {}".format(np.linalg.norm(
        python_energy_transition_path - compare_energy_transition_path
    )))
    print()
    print(">>> Displaying values \n\n")
    print(python_energy_transition_path)
    print(compare_energy_transition_path)