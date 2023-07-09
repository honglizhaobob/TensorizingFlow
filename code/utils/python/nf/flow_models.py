""" 

Use generated TT data to train for different rank choices.
Original: Hongli Zhao, 11/27/2021
Refactored: 12/27/2021

"""

# import pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

import numpy as np
from scipy import linalg as splin
import scipy.io

# very high precision
torch.set_default_dtype(torch.float64)
import torch.optim as optim

# define all flow models
class Flow(transform.Transform, nn.Module):
    
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    
    def init_parameters(self, near_identity=False):
        """ Initialize all trainable parameters (declared using 
        torch.nn.Parameter(). Provides different modes of initialization
        if needed. Default initializes near identity (slightly perturbed
        from 0). """
        for param in self.parameters():
            if not near_identity:
                # use random parameters
                param.data.uniform_(-0.01, 0.01)
            else:
                # near 0 parameters
                #param.data.normal_(0.0, 0.1)
                param.data.fill_(0)
            
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)
    
    # forward evaluation: x = f(z)
    def forward(self, z):
        pass

# Main class for normalizing flow
class NormalizingFlow(nn.Module):
    def __init__(self, dim, blocks, flow_length):
        super().__init__()
        biject = []
        for f in range(flow_length):
            if blocks is None:
                # by default uses Planar flow, which does not have inverse abiity
                biject.append(PlanarFlow(dim))
            else:
                # alternate among the blocks
                for flow in blocks:
                    biject.append(flow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det


## Planar flow
class PlanarFlow(Flow):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.init_parameters()

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs())

## Leaky ReLU Flow
class PReLUFlow(Flow):
    def __init__(self, dim):
        super(PReLUFlow, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.bijective = True

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(0.01, 0.99)

    def _call(self, z):
        return torch.where(z >= 0, z, torch.abs(self.alpha) * z)

    def _inverse(self, z):
        return torch.where(z >= 0, z, torch.abs(1. / self.alpha) * z)

    def log_abs_det_jacobian(self, z):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J))
        return torch.sum(log_abs_det, dim = 1)

## Affine transform
class AffineFlow(Flow):
    def __init__(self, dim):
        super(AffineFlow, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.orthogonal_(self.weights)

    def _call(self, z):
        return z @ self.weights
    
    def _inverse(self, z):
        return z @ torch.inverse(self.weights)

    def log_abs_det_jacobian(self, z):
        return torch.slogdet(self.weights)[-1].unsqueeze(0).repeat(z.size(0), 1)


## Affine coupling flow
class AffineCouplingFlow(Flow):
    def __init__(self, dim, n_hidden=64, n_layers=3, activation=nn.ReLU):
        super(AffineCouplingFlow, self).__init__()
        self.k = dim // 2
        self.g_mu = self.transform_net(self.k, dim - self.k, n_hidden, n_layers, activation)
        self.g_sig = self.transform_net(self.k, dim - self.k, n_hidden, n_layers, activation)
        self.init_parameters()
        self.bijective = True

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            net.append(nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden))
            net.append(activation())
        return nn.Sequential(*net)
        
    def _call(self, z):
        z_k, z_D = z[:, :self.k], z[:, self.k:]
        zp_D = z_D * torch.exp(self.g_sig(z_k)) + self.g_mu(z_k)
        return torch.cat((z_k, zp_D), dim = 1)

    def _inverse(self, z):
        zp_k, zp_D = z[:, :self.k], z[:, self.k:]
        z_D = (zp_D - self.g_mu(zp_k)) / self.g_sig(zp_k)
        return torch.cat((zp_k, z_D))

    def log_abs_det_jacobian(self, z):
        z_k = z[:, :self.k]
        return -torch.sum(torch.abs(self.g_sig(z_k)))
    
## Shuffle flow
class ReverseFlow(Flow):

    def __init__(self, dim):
        super(ReverseFlow, self).__init__()
        self.permute = torch.arange(dim-1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def _call(self, z):
        return z[:, self.permute]

    def _inverse(self, z):
        return z[:, self.inverse]

    def log_abs_det_jacobian(self, z):
        return torch.zeros(z.shape[0], 1)
    
class ShuffleFlow(ReverseFlow):

    def __init__(self, dim):
        super(ShuffleFlow, self).__init__(dim)
        self.permute = torch.randperm(dim)
        self.inverse = torch.argsort(self.permute)
    
## Batch Normalization 
class BatchNormFlow(Flow):

    def __init__(self, dim, exact_stats='gl1d', momentum=0.95, eps=1e-5):
        """ exact_stats controls whether to use exact mean and
        variance as initialization, examples are: 'gl1d', 'gl2d', 
        'rosen', 'double_rosen'. If None, use zero and ones to initialize. """
        super(BatchNormFlow, self).__init__()
        # Running batch statistics
        self.r_mean = torch.zeros(dim)
        self.r_var = torch.ones(dim)
        # Momentum
        self.momentum = momentum
        self.eps = eps
        # Trainable scale and shift (cf. original paper)
        if exact_stats == None:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            # all experiments considered
            all_exps = ['gl1d', 'gl2d', 'rosen', 'double_rosen', 'mixture', 'gl1d_anti', 'spin_glass', \
                'mix8', 'double_rosen3', 'tpt']
            assert exact_stats in all_exps, "*> Please double check input name. "
            print("::BacthNormFlow: Using exact stats: {}, for initialization. ".format(exact_stats))
            # need to make sure data is generated
            if exact_stats in ['gl1d', 'gl2d', 'rosen', 'double_rosen']:
                stats = scipy.io.loadmat('./utils/full_rank_stats.mat')
            elif exact_stats in ['mixture', 'gl1d_anti']:
                # updated
                stats = scipy.io.loadmat('./utils/full_rank_stats2.mat')
            elif exact_stats in ['spin_glass']:
                #stats = scipy.io.loadmat('./utils/full_rank_stats3.mat')
                stats = scipy.io.loadmat('./utils/full_rank_stats7.mat')
            elif exact_stats in ['mix8']:
                stats = scipy.io.loadmat('./utils/full_rank_stats4.mat')
            elif exact_stats in ['gl1d_anti']:
                stats = scipy.io.loadmat('./utils/full_rank_stats6.mat')
            elif exact_stats in ['tpt']:
                # change to 8 later
                stats = scipy.io.loadmat('./utils/full_rank_stats9.mat')
            else:
                stats = scipy.io.loadmat('./utils/full_rank_stats5.mat')
            # initialize
            self.gamma = nn.Parameter(torch.tensor(stats[exact_stats]['std'][0][0][0], \
                requires_grad=True))
            self.beta = nn.Parameter(torch.tensor(stats[exact_stats]['mean'][0][0][0], \
                requires_grad=True))
            
        
    def _call(self, z):
        if self.training:
            # Current batch stats
            self.b_mean = z.mean(0)
            self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps
            # Running mean and var
            self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - mean) / var.sqrt()
        y = self.gamma * x_hat + self.beta
        return y

    def _inverse(self, x):
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (x - self.beta) / self.gamma
        y = x_hat * var.sqrt() + mean
        return y
        
    def log_abs_det_jacobian(self, z):
        # Here we only need the variance
        mean = z.mean(0)
        var = (z - mean).pow(2).mean(0) + self.eps
        log_det = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
        return torch.sum(log_det, -1)

## Define specific classes of BatchNormFlow for each experiment
class GL1DBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(GL1DBatchNormFlow, self).__init__(dim=dim, exact_stats='gl1d')

class GL2DBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(GL2DBatchNormFlow, self).__init__(dim=dim, exact_stats='gl2d')

class RosenBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(RosenBatchNormFlow, self).__init__(dim=dim, exact_stats='rosen')

class DoubleRosenBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(DoubleRosenBatchNormFlow, self).__init__(dim=dim, exact_stats='double_rosen')

class GL1DAntiBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(GL1DAntiBatchNormFlow, self).__init__(dim=dim, exact_stats='gl1d_anti')

class MixtureGaussianBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(MixtureGaussianBatchNormFlow, self).__init__(dim=dim, exact_stats='mixture')

class SpinGlassBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(SpinGlassBatchNormFlow, self).__init__(dim=dim, exact_stats='spin_glass')

class MixtureGaussianBatchNormFlow8Mode(BatchNormFlow):
    def __init__(self, dim):
        super(MixtureGaussianBatchNormFlow8Mode, self).__init__(dim=dim, exact_stats='mix8')

class DoubleRosenBatchNormFlow3Dim(BatchNormFlow):
    def __init__(self, dim):
        super(DoubleRosenBatchNormFlow3Dim, self).__init__(dim=dim, exact_stats='double_rosen3')

class PathSamplingBatchNormFlow(BatchNormFlow):
    def __init__(self, dim):
        super(PathSamplingBatchNormFlow, self).__init__(dim=dim, exact_stats='tpt')

class MaskedLinearAR(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(MaskedLinearAR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.init_parameters()

    def init_parameters(self, ):
        # near identity (weights near 0)
        nn.init.xavier_normal_(self.weight.data)

        # near identity (bias near 0)
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        output = input @ self.weight.tril(-1)
        output += self.bias
        return output

## Masked Autoregressive Flow
class MAFlow(Flow):
    def __init__(self, dim, n_hidden=100, n_layers=3, activation=nn.ReLU):
        super(MAFlow, self).__init__()
        self.g_mu = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.g_sig = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.init_parameters()
        self.bijective = True

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            net.append(MaskedLinearAR(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden))
            net.append(activation())
        return nn.Sequential(*net)
        
    def _call(self, z):
        zp = z * torch.exp(self.g_sig(z)) + self.g_mu(z)
        return zp

    def _inverse(self, z):
        z = (z - self.g_mu(z)) / self.g_sig(z)
        return z

    def log_abs_det_jacobian(self, z):
        return -torch.sum(torch.abs(self.g_sig(z)))

## Naive Implementation of ResNet (https://arxiv.org/pdf/1908.09257.pdf)
class NaiveResidualFlow(Flow):
    """
        Residual flow of the form (naive implementation without coupling):
                        z_k = z_0 + F(z_0)
        where F is a feed-forward network of any kind (default MLP). 

        Inverse is not implemented (does not have simple inverse).
    """
    def __init__(self, dim, jacobian_estimator='exact', 
                    trace_series_truncation_level=500, 
                    n_hidden=64, 
                    n_layers=3, 
                    activation=nn.ReLU):
        super(NaiveResidualFlow, self).__init__()
        # define neural net
        self._F_net = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.init_parameters()
        # estimator used for log det Jacobian evaluation
        self.dim = dim
        self._trace_series_truncation_level = trace_series_truncation_level
        if jacobian_estimator == 'basic':
            self._estimator = self._basic_jac_estimator
        if jacobian_estimator == 'hutchinson':
            self._estimator = self._hutchinson_jac_estimator
        if jacobian_estimator == 'exact':
            self._estimator = self._exact_jac


    
    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            net.append(nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden))
            net.append(activation())
        return nn.Sequential(*net)

    def _call(self, z):
        """ forward evaluation. """
        return z + self._F_net(z)

    def _inverse(self, z):
        raise NotImplementedError()

    def log_abs_det_jacobian(self, z):
        """ according to the paper, log jacobian is not easily evaluated, 
        must resort to approximations. Defaults to 'basic' estimator, which
        uses exact evaluation of trace. 

        *(warning) Lipschitz constant of F must be < 1, implicitly assumed.
        """
        batch_result = self._estimator(z)
        return batch_result.reshape(-1, 1)

    # helper method for jacobian calculation 
    def _exact_jac(self, z):
        """ exact computation of log det Jacobian, may be very slow. """
        z = z.requires_grad_(True)
        batch_jacobians = self._batch_jacobian(self._F_net(z), z)
        # add identity to each Jacobian and take determinant
        batch_result = torch.log(torch.abs(torch.det(batch_jacobians + torch.eye(self.dim))))
        return batch_result

    # helper methods for jacobian estimation
    def _basic_jac_estimator(self, z):
        """ uses truncated series and exact trace evaluation. """
        z = z.requires_grad_(True)

        # apply NN on z, compute Jacobian traces
        all_traces = self._batch_trace_basic(self._batch_jacobian(self._F_net(z), z))

        batch_jacobians = self._batch_jacobian(self._F_net(z), z)
        print(batch_jacobians.shape)


        # create all indices 
        k = torch.range(1, self._trace_series_truncation_level)
        # create coefficients (-1)**(k+1) / k
        coefs = torch.Tensor([-1])**(k+1) / k
        # compute all powers: size (b x k)
        trace_pwr_k = all_traces.reshape(-1, 1) ** k
        # return batch result
        batch_result = coefs.reshape(1, -1) @ trace_pwr_k.T
        print(torch.norm(batch_result.reshape(-1, 1)))
        return batch_result

    def _batch_jacobian(self, y, x):
        """ x is a minibatch of data (b x d), computes Jacobian of y = F(x) at 
        each data point in the batch. Assumes requires_grad == True. Returns a
        (list) of (torch.Tensor), of size (d x d) each. """
        jac = []
        for d in range(y.shape[1]):
            jac.append(torch.autograd.grad(torch.sum(y[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
        return torch.cat(jac, 1)

    def _batch_trace_basic(self, batch_jac):
        """ Computes trace exactly on a (list) of Jacobians. """
        M = batch_jac
        return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)

    def _hutchinson_jac_estimator(self):
        """ uses truncated series with Hutchinson Monte-Carlo estimate for trace. 
        (Memory efficient). """
        raise NotImplementedError()
    
    def _batch_trace_hutch(self, batch_jac):
        """ Computes trace on a (list) of Jacobians via the Hutchinson Monte-Carlo method. """
        raise NotImplementedError()

## Example specific resnet
# GL 1d
class GL1DResidualFlow(NaiveResidualFlow):
    def __init__(self, dim):
        super(GL1DResidualFlow, self).__init__(dim=dim,\
             trace_series_truncation_level=1500, \
                 n_hidden=128, \
                     n_layers=4)

# Mixture Gaussian with 5 modes: potentially needs powerful residual flow
class MixtureGaussianResidualFlow(NaiveResidualFlow):
    def __init__(self, dim):
        super(MixtureGaussianResidualFlow, self).__init__(dim=dim,\
             trace_series_truncation_level=1000, \
                 n_hidden=32, \
                     n_layers=5)

# GL1D Spin Glass
class SpinGlassResidualFlow(NaiveResidualFlow):
    def __init__(self, dim):
        super(SpinGlassResidualFlow, self).__init__(dim=dim,\
             trace_series_truncation_level=1000, \
                 n_hidden=128, \
                     n_layers=3)

# Nonequilibrium transition path sampling
class PathSamplingResidualFlow(NaiveResidualFlow):
    def __init__(self, dim):
        super(PathSamplingResidualFlow, self).__init__(dim=dim,\
             trace_series_truncation_level=1000, \
                 n_hidden=150, \
                     n_layers=4)



## Example specific blocks

REAL_NVP_BLOCKS = [ AffineCouplingFlow, ReverseFlow, BatchNormFlow ]
AUTOREGRESSIVE_BLOCKS = [ MAFlow, ReverseFlow, BatchNormFlow ]
LEAKY_MLP_BLOCKS = [ AffineFlow, BatchNormFlow, PReLUFlow ]
RESNET_BLOCKS = [ NaiveResidualFlow, ReverseFlow, BatchNormFlow ]
RESNET_BLOCKS2 = [ NaiveResidualFlow, BatchNormFlow ]

## Experiment specific (RealNVP)
REALNVP_BLOCKS_GL1D = [ AffineCouplingFlow, ReverseFlow, BatchNormFlow ]

## Experiment specific (ResNet)
#RESNET_BLOCKS2_GL1D = [ GL1DResidualFlow, GL1DBatchNormFlow ]
RESNET_BLOCKS2_GL1D = [ NaiveResidualFlow, GL1DBatchNormFlow ]
RESNET_BLOCKS2_GL2D = [ NaiveResidualFlow, GL2DBatchNormFlow ]
RESNET_BLOCKS2_ROSEN = [ NaiveResidualFlow, RosenBatchNormFlow ]
RESNET_BLOCKS2_DROSEN = [ NaiveResidualFlow, DoubleRosenBatchNormFlow ]
RESNET_BLOCKS2_GL1D_ANTI = [ NaiveResidualFlow, GL1DAntiBatchNormFlow ]
RESNET_BLOCKS2_MIXTURE_GAUSSIAN = [ MixtureGaussianResidualFlow, MixtureGaussianBatchNormFlow ]
RESNET_BLOCKS2_SPIN_GLASS = [ SpinGlassResidualFlow, SpinGlassBatchNormFlow ]
RESNET_BLOCKS2_MIXTURE_GAUSSIAN_8MODE = [ NaiveResidualFlow, MixtureGaussianBatchNormFlow8Mode ]
RESNET_BLOCKS2_DROSEN3 = [ NaiveResidualFlow, DoubleRosenBatchNormFlow3Dim ]
RESNET_PATH = [ PathSamplingResidualFlow, PathSamplingBatchNormFlow ]
REAL_NVP_BLOCKS_PATH = [ AffineCouplingFlow, ReverseFlow, PathSamplingBatchNormFlow ]