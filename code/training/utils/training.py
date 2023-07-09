""" Utility functions for training normalizing flow models.

# Refactored: 12/31/2021
"""

# import pytorch
from tabnanny import check
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

# numpy
import numpy as np

# very high precision
torch.set_default_dtype(torch.float64)
import torch.optim as optim

# set number of threads (8 threads should work on most computers)
torch.set_num_threads(8)

# supress warnings
import warnings
warnings.filterwarnings("ignore")

TF_VISUAL_REPORT = True
# import matplotlib if avaiable
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("utils::flow_models: Matplotlib is unavailable on this machine. ")
    TF_VISUAL_REPORT = False

def train(dataset, model, target, 
          num_epochs=150,
          batch_size=2**7,
          verbose=True,
          lr=1e-3, 
          use_scheduler=False,
          schedule_rate=0.9999,
          grad_clip=1e+4, 
          plot_it=TF_VISUAL_REPORT):
    """"
        Train a normalizing flow to bridge gap between prior and target. Loss function is 
    KL divergence between posterior (TT + NF) and target (analytic expression). All target
    distributions assume the Boltzmann equilibrium form: 
        \pi(x) \sim \exp(-beta * E(x)) where E is energy,
        Log target probability density is directly input as -beta * E(x) for stable evaluation.

    Args:
        dataset (TensorizingFlowDataset) contains training and test data, with log pdf values.

        model  (flow_model)
        target (function)                specify target distribution, supported examples in
                                         target.py
                                         - ginzburg_landau1d_logpdf
                                         - ginzburg_landau2d_logpdf
                                         - rosen_logpdf
                                         - double_rosen_logpdf
        dataloader (torch.utils.DataLoader) 
        num_epochs
        batch_size
        verbose (bool)
        lr (float): learning rate
        use_scheduler (bool): if learning rate schedule should be used, default to ExponentialScheduler
        step_schedule (int):  learning rate exponential decay momentum

        plot_it       (bool)            Plot the samples or not, must set to False
                                         if matplotlib is unavailable.
    """
    #assert num_epochs >= 10, "Please set a larger number of epochs for sufficient convergence. "
    # customize KL divergence loss function
    def loss_func(x, log_jacobians, prior_logpdf, targ_logpdf=target, plotit=False):
        """ evaluate KL divergence between posterior distribution (NF + prior) 
        and target. x is flowed samples. If plotit is set to True, plots histograms 
        of 2 categories of values: sum of log jacobians, target log pdf. Only use plotit
        for debugging. """
        sum_of_log_jacobians = sum(log_jacobians)
        if plotit and x.shape[0] > 5000:
            plt.figure(10)
            #plt.hist(np.array([jacob.detach().numpy() for jacob in log_jacobians]))
            print("Sum of log jacobians = {}".format(sum_of_log_jacobians.detach().numpy()))
            plt.figure(11)
            plt.hist(targ_logpdf(x).detach().numpy())
            plt.show()
        return (prior_logpdf - sum_of_log_jacobians - targ_logpdf(x)).mean()
    
    def loss_func_init(x, prior_logpdf, targ_logpdf=target):
        """ evaluate initial KL divergence between posterior distribution (NF + prior) 
        and target. x is samples without flow. This is a Monte-Carlo estimation of the 
        log partition function. """
        return (prior_logpdf - targ_logpdf(x)).mean()
    
    # save hyperparameters before training
    initial_learning_rate = lr

    # before training, report and save the initial KL divergence
    z0_test, z0_test_logpdf = dataset.get_test_data()
    loss_initial = loss_func_init(z0_test, z0_test_logpdf).item()
    if verbose:
        print("[ Before Training ]:: ( KL-Divergence ) = {}".format(loss_initial))

    if plot_it:
        plot_zk_test = z0_test.detach().numpy()
        try:
            # plot last two dimensions
            plt.figure(figsize=(6, 4))
            plt.hexbin(plot_zk_test[:,-2], plot_zk_test[:,-1], cmap='rainbow', gridsize=(120,120))
            plt.grid(True)
            plt.xlabel('Dimension {}'.format(dataset.dim-1))
            plt.ylabel('Dimension {}'.format(dataset.dim))
            plt.title("Unflowed Samples")
            plt.show()
        except ImportError:
            print("utils::training::train: Matplotlib is unavailable on this machine. Setting plot_it to False. ")
            plot_it = False

    # initialize training model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, schedule_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        #                                            step_size=step_schedule, 
        #                                            gamma=0.5)
    
    # record preallocate
    all_training_loss = np.zeros([num_epochs])      # training loss averaged over each epoch
    all_test_loss = np.zeros([num_epochs])          # generalization loss computed on all test data at once
    model_snapshots = [model.state_dict()]          # snapshots of NF recorded each epoch
    all_grad_norms = np.zeros([num_epochs])         # record norm of gradients

    # save all flowed samples (from test dataset)
    all_flowed_samples = np.zeros([num_epochs, z0_test.shape[0], z0_test.shape[1]])

    epoch = 0
    # number of allowance to redo one epoch
    patience = 0
    while epoch < num_epochs:

        # get torch dataloader for each epoch
        tensorizing_dataloader = dataset.get_dataloader(batch_size=batch_size)
        # accumulate loss over batch (used for computing average)
        acc_train_loss = []

        # save model, optimizer state for retraining
        save_model_state = model.state_dict()
        save_optimizer_state = optimizer.state_dict()

        # iterate over each batch
        for batch_number, minibatch in enumerate(tensorizing_dataloader):
            # samples before flowing
            z0 = minibatch[0]
            # prior logpdf
            z0_logpdf = minibatch[1]
            # apply Normalizing Flow 
            zk, log_jacobians = model(z0)

            # clean optimizer for backward()
            optimizer.zero_grad()
            # compute minibatch sample KL divergence
            loss_train = loss_func(zk, log_jacobians, z0_logpdf)
            # backprop
            loss_train.backward()

            # check if training loss blew up, if it did, do not take step
            # namely that batch is wasted/discarded because it was "unlucky"
            if loss_train.item() <= 1e+5:
                # take a step

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                # update optimizer
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                # accumulate training loss per epoch
                acc_train_loss.append(loss_train.item())
                if verbose and (batch_number % 10 == 0):
                    print('[ #### ]:: In training ...  ... (epoch{}=>batch{})'.format(epoch+1, batch_number))
            else:
                # do not take a step
                print('[ #### ]:: Training encountered blowup: {} ...  ... (epoch{}=>batch{}) discarded! '\
                .format(loss_train.item(), \
                epoch+1, batch_number))
                continue
 
        ## Reporting after each epoch

        # get test data
        z0_test, z0_test_logpdf = dataset.get_test_data()

        # use current NF model and flow all test samples
        zk_test, log_jacobians_test = model(z0_test)

        # compute generalization err. on all test data at once
        loss_test = loss_func(zk_test, log_jacobians_test, z0_test_logpdf).item()

        # compute averaged training error
        acc_train_loss = np.mean(acc_train_loss)        

        # check for NaN loss value and stop training
        if np.isinf(loss_test) or np.isnan(loss_test) or np.isnan(acc_train_loss):
            print('Training stopped because loss became Inf or NaN!')

            return {'model_snapshots': model_snapshots, 
                    'training_loss': all_training_loss,
                    'test_loss': all_test_loss,
                    'grad_norms': all_grad_norms,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'lr': initial_learning_rate,
                    'post_training_samples': zk_test.detach().numpy(),
                    'message': 'Training stopped because loss became Inf or NaN!'
                    }
        # should not do more than 20 times per epoch
        if patience >= 20:
            print('Terminating as patience is exceeded ... tune hyperparameters instead. ')
            return {'model_snapshots': model_snapshots, 
                    'training_loss': all_training_loss,
                    'test_loss': all_test_loss,
                    'grad_norms': all_grad_norms,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'lr': initial_learning_rate,
                    'post_training_samples': zk_test.detach().numpy(),
                    'message': 'Terminating as patience is exceeded ... tune hyperparameters instead. '
                    }

        check_blow_up = True

        if check_blow_up:
            # check for test loss blowing up; should not blow up to more than initial loss
            if loss_test > 1e+3:
                print('Reverting (epoch{}) because test error blew up: {} ... '.format(epoch+1, loss_test))
                # revert model state back to before this epoch
                # implicitly, because we didn't save learning rate scheduler state
                # this means we retrain this epoch with a smaller learning rate
                model.load_state_dict(save_model_state)
                optimizer.load_state_dict(save_optimizer_state)
                optimizer.zero_grad()
                # increment patience
                patience += 1
                continue

        # record after epoch training
        all_training_loss[epoch] = acc_train_loss
        all_test_loss[epoch] = loss_test
        all_flowed_samples[epoch, :, :] = zk_test.detach().numpy()

        # save a snapshot of the model after each epoch
        model_snapshots.append(model.state_dict())
        #for p in model.parameters():
        #    print(p.grad)
        # record norm of gradients as sanity check (exploding gradient)
        total_grad_norm = np.sum([( p.grad.detach().data.norm(2) ) ** 2 for p in model.parameters() if p.grad is not None])
        all_grad_norms[epoch] = total_grad_norm

        # visual reporting (if matplotlib is available)
        if plot_it and (epoch % 5 == 0):
            plot_zk_test = zk_test.detach().numpy()
            try:
                # 2 dimensions case
                if dataset.dim == 2:
                    # only 1 plot to show
                    plt.figure(1, figsize=(6, 4))
                    plt.hexbin(plot_zk_test[:,0], plot_zk_test[:,1], cmap='rainbow', gridsize=(120,120))
                    plt.grid(True)
                    plt.title("Flowed Samples at: Epoch {}".format(epoch+1))
                    plt.show()
                # 3 dimensions case
                if dataset.dim == 3:
                    # only 2 plots to show
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    ax[0].hexbin(plot_zk_test[:,0], plot_zk_test[:,1], cmap='rainbow', gridsize=(120,120))
                    ax[0].grid(True)
                    ax[1].hexbin(plot_zk_test[:,1], plot_zk_test[:,2], cmap='rainbow', gridsize=(120,120))
                    ax[1].grid(True)
                    plt.suptitle("Flowed Samples at: Epoch {}".format(epoch+1))
                    plt.show()
                # >=3 dimensions
                if dataset.dim > 3:
                    # plot last two dimensions
                    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
                    ax[0].hexbin(plot_zk_test[:,-4], plot_zk_test[:,-3], cmap='rainbow', gridsize=(120,120))
                    ax[0].grid(True)
                    #ax[0].set_xlim([-1.5, 1.5])
                    #ax[0].set_ylim([-1.5, 1.5])
                    ax[1].hexbin(plot_zk_test[:,-3], plot_zk_test[:,-2], cmap='rainbow', gridsize=(120,120))
                    ax[1].grid(True)
                    #ax[1].set_xlim([-1.5, 1.5])
                    #ax[1].set_ylim([-1.5, 1.5])
                    ax[2].hexbin(plot_zk_test[:,-2], plot_zk_test[:,-1], cmap='rainbow', gridsize=(120,120))
                    ax[2].grid(True)
                    #ax[2].set_xlim([-1.5, 1.5])
                    #ax[2].set_ylim([-1.5, 1.5])
                    plt.suptitle("Flowed Samples at: Epoch {}".format(epoch+1))
                    plt.show()
            except ImportError:
                print("utils::training::train: Matplotlib is unavailable on this machine. Setting plot_it to False. ")
                plot_it = False
        
        # report training progress
        if verbose:
            print("[ Epoch  {} ]:: ( Train Avg. Over Epoch ) = {}, ( Generalization ) = {}"\
                .format(epoch + 1, acc_train_loss, loss_test))
            if use_scheduler:
                print('[ ======== ]:: Report Learning Rate =  {}'.format(optimizer.param_groups[0]['lr']))
            print('[ ======== ]:: Report Norm of Gradient =  {}'.format(total_grad_norm))
        
        # check next epoch
        epoch += 1
        # reset patience
        patience = 0

        
    ## After all epochs, export data 
    
    # use final NF model to flow the test samples and save
    z0_test, _ = dataset.get_test_data()
    zk_last, _ = model(z0_test)
    zk_last = zk_last.detach().numpy()

    if verbose:
        print('[ ========= ]:: Tensorizing Flow training finished, last flow copy saved locally. ')

    if plot_it:
        # plot loss curves
        plt.figure(figsize=(12, 6))
        plt.plot(range(num_epochs), all_training_loss, label="( Training Loss )")
        plt.plot(range(num_epochs), all_test_loss, label="( Test Loss )")
        plt.axhline(y=loss_initial, linestyle='-.', color='purple', label='( Initial Loss )')
        plt.xlabel("( Epoch # )")
        plt.ylabel("( Loss )")
        plt.title("Tensorizing Flow Loss Profile: Num Epoch {}, Batch Size {}".format(num_epochs, batch_size))
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

    # save all necessary reports
    to_return = {
        'initial_loss': loss_initial,
        'model_snapshots': model_snapshots,
        'training_loss': all_training_loss,
        'test_loss':all_test_loss,
        'post_training_samples': zk_last,
        'grad_norms': all_grad_norms,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': initial_learning_rate,
        'message': 'Success. '
    }
    return to_return