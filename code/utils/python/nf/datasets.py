
""" 
Utility functions for loading data from MATLAB and creating
batched PyTorch datasets.

# Refactored: 12/28/2021
"""

# import pytorch dataloader
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import scipy
import scipy.io
import numpy as np

# supress warnings
import warnings
warnings.filterwarnings("ignore")

## Dataset classes for PyTorch training
class TensorizingFlowDataset(Dataset):
    """ Constructs the dataset by loading MAT_FILE, assuming
    it is stored under ./DATA/, checks for existence of data. 
    For details, check MATLAB data generation schemes.
    """
    def __init__(self, mat_file, root_dir=os.getcwd() + "/data/", gaussian_data=False, requires_grad=False):
        """ If Gaussian data is True, uses the Gaussian samples attached with the 
        .MAT dataset (if full rank dataset, will be exact Gaussian; if not, will be N(0, 0.05) Gaussian). 
            * If requires_grad == True, tensor will keep record of gradient for all data (may be expensive).
        """
        self.root_dir = root_dir
        # checks sample folder exists
        assert os.path.isdir(self.root_dir), "Please make sure sample folder is created. "
        self.mat_filename = self.root_dir + mat_file
        assert os.path.isfile(self.mat_filename), "Please make sure samples are generated. "
        # load .mat data
        self.raw_dataset = scipy.io.loadmat(self.mat_filename)

        # refactor all variables
        if not gaussian_data:
            # load samples
            self.training_data = self.raw_dataset['X_train'].T
            self.test_data = self.raw_dataset['X_test'].T
            # load corresponding log-likelihoods
            self.log_training_data_densities = np.log(self.raw_dataset['likes_train'])
            self.log_test_data_densities = np.log(self.raw_dataset['likes_test'])
        else:
            # load samples
            self.training_data = self.raw_dataset['X_bad_train'].T
            self.test_data = self.raw_dataset['X_bad_test'].T
            # load corresponding log-likelihoods
            self.log_training_data_densities = np.log(self.raw_dataset['likes_bad_train'])
            self.log_test_data_densities = np.log(self.raw_dataset['likes_bad_test'])
        
        # convert to torch.Tensor (requires grad optional)
        self.training_data = torch.tensor(self.training_data, requires_grad=requires_grad)
        self.test_data = torch.tensor(self.test_data, requires_grad=requires_grad)
        self.log_training_data_densities = torch.tensor(self.log_training_data_densities, requires_grad=requires_grad)
        self.log_test_data_densities = torch.tensor(self.log_test_data_densities, requires_grad=requires_grad)
        # hyperparameters
        self.num_samples = self.training_data.shape[0] # number of samples assumes 50-50 splitting
        self.dim = self.training_data.shape[1]

    def __len__(self):
        """ return dataset size. """
        return self.num_samples
    
    def __getitem__(self, idx):
        """ slice into data based on indices in IDX. """
        samples = self.training_data[idx, :]
        samples_log_probs = self.log_training_data_densities[idx, :].reshape(-1, 1)
        return (samples, samples_log_probs)

    def get_test_data(self):
        """ query all test data for validation. """
        return (self.test_data, self.log_test_data_densities)

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, drop_last=True):
        """ Creates a PyTorch DataLoader with preferred parameters.
        This function is a wrapper. See: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        for a detailed explanation of parameters. 
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    
