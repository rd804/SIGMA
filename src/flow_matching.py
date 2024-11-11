from multiprocessing import context
from pyexpat import model
#import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
import time
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from zuko.utils import odeint
import torch.nn as nn
# from nflows.nn.nets import ResidualNet
import copy

from torch.distributions import Normal
from sklearn.metrics import roc_curve, roc_auc_score
import os
import numpy as np
import torch

def log_normal(x,mu=0.0,sigma=1.0):
    return Normal(mu, sigma).log_prob(x).sum(-1)

def log_prob(model: torch.nn.Module , x: Tensor, context: Tensor,
             start:float=0.0, end:int=1.0) -> Tensor:
        # get the identity matrix with shape (batch_size, features, features)
        identity = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        # move the last dimension to the first dimension
        identity = identity.expand(*x.shape, x.shape[-1]).movedim(-1, 0)


        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:

            with torch.enable_grad():
                x = x.requires_grad_()
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                #print(t)
                #input_to_model = torch.cat([x,t_array], dim=-1)
            #    print(input_to_model.shape)
            #    print(context.shape)
                vt = model(input_to_model, context=context)

            #print(vt.shape)

            jacobian = torch.autograd.grad(
                vt, x, identity, create_graph=True, is_grads_batched=True
            )[0]
            # calculate the trace of the jacobian
            trace = torch.einsum("i...i", jacobian)


          #  print(trace)

            return vt, trace * 1e1  # 1e-2 is a scaling factor for numerical stability

        # initial value for the log_abs_det_jacobian (the subsequent contributions
        # are added to this value in the odeint call below)
        ladj = torch.zeros_like(x[..., 0])
        # integrate the augmented function from t=0 to t=1
        # --> returns the latent space z and the integral over the trace of the
        #     jacobian

        z, ladj = odeint(augmented, (x, ladj), start, end, phi=model.parameters())

        return  z, log_normal(z) + ladj * 1e-1  # rescale the ladj to undo the scaling above

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model: nn.Module, context: None = None):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t, x_, *args, **kwargs):
        x = x_[:, :-1]
        identity = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        identity = identity.expand(*x.shape, x.shape[-1]).movedim(-1, 0)

      #  print(t.shape)
        with torch.enable_grad():
            x = x.requires_grad_()
            self.model.eval()
            x_input = torch.cat([x, t.repeat(x.shape[0])[:, None]], 1)
            v_input = self.model(x_input, context=self.context)

            jacobian = torch.autograd.grad(v_input, x, identity, 
                        create_graph=True, is_grads_batched=True)[0]
                # calculate the trace of the jacobian
            trace = torch.einsum("i...i", jacobian)

        output = torch.cat([v_input, trace.reshape(-1,1) ], dim=1)
        
        return output

class torch_wrapper_sample(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model: nn.Module, context: None = None):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t, x_, *args, **kwargs):
        x = x_

        x_input = torch.cat([x, t.repeat(x.shape[0])[:, None]], 1)
        with torch.no_grad():
            self.model.eval()
            v_t = self.model(x_input, context=self.context)


        
        return v_t

def sample_torchdyn(model, x, device,
                      start = 0.0, end=1.0, intervals=2):
    
    ode_model = torch_wrapper_sample(model, context=x[:,0].reshape(-1,1))
    x_input = x[:,1:]
    node = NeuralODE(ode_model, solver="dopri5", 
         sensitivity="adjoint", atol=1e-4, rtol=1e-4).to(device)
    
    t_span = torch.linspace(start,end, intervals)
    t_eval, trajectory = node(x_input, t_span)

    trajectory = trajectory.detach().cpu()

    return trajectory[-1]


def log_prob_torchdyn(model, x, device,
                      start = 1.0, end=0.0, intervals=2,
                      train=False):
    ode_model = torch_wrapper(model, context=x[:,0].reshape(-1,1))
    ladj = torch.zeros_like(x[..., 0]).reshape(-1,1)
# print(ladj.shape)
    x_input = torch.cat([x[:,1:], ladj], dim=1)
#   print(x_input.shape)
    node = NeuralODE(ode_model, solver="dopri5", 
         sensitivity="adjoint", atol=1e-4, rtol=1e-4).to(device)

    t_span = torch.linspace(start,end, intervals)
    t_eval, trajectory = node(x_input, t_span)
    if not train:
        trajectory = trajectory.detach().cpu()

    latent_space = trajectory[-1,:,:-1]
    log_probability = log_normal(latent_space) + trajectory[-1,:,-1]

    return trajectory, log_probability


    


def SIC(label, score):
    fpr, tpr, thresholds = roc_curve(label, score)
    auc = roc_auc_score(label, score)

    tpr = tpr[fpr>0]
    fpr = fpr[fpr>0]

    sic = tpr/np.sqrt(fpr)

    return sic, tpr, auc

def prob_x(x,mean,sigma):
    return np.exp(Normal(mean, sigma).log_prob(x).sum(-1))
    
back_mean = 0
back_sigma = 3
sig_mean = 2
sig_sigma = 0.25
# n_bkg = 120_000
# n_sig = 2000
# w = n_sig / (n_bkg + n_sig)        

def logp_bkg(x):
    return Normal(back_mean, back_sigma).log_prob(x).sum(-1)

def logp_sig(x):
    return Normal(sig_mean, sig_sigma).log_prob(x).sum(-1)

def logp_data(x,w):
    return torch.log(w * np.exp(logp_sig(x)) + (1 - w) * np.exp(logp_bkg(x)))

def anom_score(x):
    return np.exp(logp_data(x) - logp_bkg(x))




def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    #t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    #xt = t * x1 + (1 + (sigma-1) * t) * x0
    mu_t = t * x1
   # epsilon = torch.randn_like(x0)
    sigma_t = 1-(1-sigma)*t

    return mu_t + sigma_t * x0

def compute_conditional_vector_field(x0, x1, sigma):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    #return x1 - x0
    return x1 - (1-sigma) * x0


def create_loader(data, shuffle=True):
    if shuffle:
        shuffled_indices = torch.randperm(data.shape[0])
        data_shuffled = data[shuffled_indices]
        
        return data_shuffled

from torch.nn import init
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        non_linear_context=False
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
            # self.context_layer = nn.Sequential(nn.Linear(context_features, features),
            #                                     nn.BatchNorm1d(features, eps=1e-3),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(p=dropout_probability),
            #                                     nn.Linear(features,features))
                                               
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)
        self.non_linear_context = non_linear_context

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            context = self.context_layer(context)
            if self.non_linear_context:
                context = self.activation(context)
        #    context = self.activation(context)
            temps = F.glu(torch.cat((temps, context), dim=1), dim=1)

        return inputs + temps

class ResidualBlock_linear_interpolation(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

            # self.context_layer = nn.Sequential(nn.Linear(context_features, features),
            #                                     nn.BatchNorm1d(features, eps=1e-3),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(p=dropout_probability),
            #                                     nn.Linear(features,features))
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context1=None, context2=None, weight=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context1 is not None:
            context1 = self.context_layer(context1)
            context1 = self.activation(context1)
            context2 = self.context_layer(context2)
            context2 = self.activation(context2)

            context = weight * context1 + (1 - weight) * context2
            temps = F.glu(torch.cat((temps, context), dim=1), dim=1)

        return inputs + temps




class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        non_linear_context=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.non_linear_context = non_linear_context

        if context_features is not None:
            # self.initial_layer = nn.Sequential(nn.Linear(in_features + context_features, hidden_features),
            #                                     nn.BatchNorm1d(hidden_features, eps=1e-3),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(p=dropout_probability),
            #                                     nn.Linear(hidden_features,hidden_features))
           self.initial_layer = nn.Linear(
               in_features + context_features, hidden_features
           )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    non_linear_context=non_linear_context
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.activation = activation

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
            if self.non_linear_context:
                temps = self.activation(temps)
        #    temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

class ResidualNet_linear_interpolation(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock_linear_interpolation(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.activation = activation

    def forward(self, inputs, context1=None, context2=None, weight=None):
        if context1 is None:
            temps = self.initial_layer(inputs)
        else:
            temps1 = self.initial_layer(torch.cat((inputs, context1), dim=1))
            temps2 = self.initial_layer(torch.cat((inputs, context2), dim=1))
            temps1 = self.activation(temps1)
            temps2 = self.activation(temps2)
            temps = weight * temps1 + (1 - weight) * temps2
        for block in self.blocks:
            temps = block(temps, context1=context1, context2=context2, weight=weight)
        outputs = self.final_layer(temps)
        return outputs

class ResidualNet_linear_interpolation_block(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        non_linear_context=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    non_linear_context=non_linear_context
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.activation = activation

    def forward(self, inputs, context1=None, context2=None, weight=None):
        if context1 is None:
            temps = self.initial_layer(inputs)
        else:
            temps1 = self.initial_layer(torch.cat((inputs, context1), dim=1))
            temps2 = self.initial_layer(torch.cat((inputs, context2), dim=1))
            temps1 = self.activation(temps1)
            temps2 = self.activation(temps2)
            temps = weight * temps1 + (1 - weight) * temps2
        for k, block in enumerate(self.blocks):
            temps1 = block(temps, context=context1)
            temps2 = block(temps, context=context2)
            temps = weight * temps1 + (1 - weight) * temps2
            #temps = block(temps, context1=context1, context2=context2, weight=weight)
        outputs = self.final_layer(temps)
        return outputs


class ResidualNet_linear_interpolation_indiv_block(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        non_linear_context=False,
        interp_block=2
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    non_linear_context=non_linear_context
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.activation = activation
        self.interp_block = interp_block
        self.non_linear_context = non_linear_context

    def forward(self, inputs, context1=None, context2=None, weight=None, context=None):
        if context1 is None:
            temps = self.initial_layer(inputs)
        else:
            if 0 == self.interp_block:
                temps1 = self.initial_layer(torch.cat((inputs, context1), dim=1))
                temps2 = self.initial_layer(torch.cat((inputs, context2), dim=1))
                temps1 = self.activation(temps1)
                temps2 = self.activation(temps2)
                temps = weight * temps1 + (1 - weight) * temps2
            else:
                temps = self.initial_layer(torch.cat((inputs, context), dim=1))
                if self.non_linear_context:
                    temps = self.activation(temps)

        for k, block in enumerate(self.blocks):
            block_index = k+1
            if block_index == self.interp_block:
                temps1 = block(temps, context=context1)
                temps2 = block(temps, context=context2)
                temps = weight * temps1 + (1 - weight) * temps2
            else:
                temps = block(temps, context=context)

            # temps = block(temps, context1=context1, context2=context2, weight=weight)
        outputs = self.final_layer(temps)
        return outputs




class Conditional_ResNet_low_freq(nn.Module):
    def __init__(self, context_frequencies: int=3, time_frequencies=3,context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
            #self.context_dim = 2*frequencies + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context = self.context_frequencies * context[...,None]
            context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

            # set frequencies higher than 2 to zero
            context[:,3:len(self.context_frequencies)] = 0.0
            context[:,3+len(self.context_frequencies):] = 0.0

        if context is not None:
            context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context=context)


class Conditional_ResNet(nn.Module):
    def __init__(self, context_frequencies: int=3, time_frequencies=3,context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
            #self.context_dim = 2*frequencies + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context = self.context_frequencies * context[...,None]
            context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context is not None:
            context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context=context)


class Conditional_MLP(nn.Module):
    def __init__(self, context_frequencies: int=3, time_frequencies=3,context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256):
        super().__init__()


        self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
        self.time_frequencies = self.time_frequencies.to(device)
        self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
        self.context_frequencies = self.context_frequencies.to(device)
        #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
        #self.time_frequencies = self.time_frequencies.to(device)
        self.context_dim = 2*context_frequencies + 2*time_frequencies
            #self.context_dim = 2*frequencies + context_features

        self.hidden_dim = hidden_dim

        mlp_input_dim = input_dim + self.context_dim

        self.model = nn.Sequential(nn.Linear(mlp_input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, input_dim)).to(device)

    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context = context.flatten()

       # if self.time_embed is not None:
        #    t = self.time_embed(_t)
        #else:
        t = self.time_frequencies * _t[...,None]
        t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
        context = self.context_frequencies * context[...,None]
        context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context is not None:
            context = torch.cat((t, context), dim=-1)
        else:
            context = t
        
        # concatenate _x and context
        x = torch.cat((_x, context), dim=-1)

        return self.model(x)


class Conditional_MLP_linear_interpolation(nn.Module):
    def __init__(self, context_frequencies: int=3, time_frequencies=3,context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256):
        super().__init__()

        self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
        self.time_frequencies = self.time_frequencies.to(device)
        self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
        self.context_frequencies = self.context_frequencies.to(device)
        #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
        #self.time_frequencies = self.time_frequencies.to(device)
        self.context_dim = 2*context_frequencies + 2*time_frequencies
            #self.context_dim = 2*frequencies + context_features

        self.hidden_dim = hidden_dim

        mlp_input_dim = input_dim + self.context_dim

        self.model = nn.ModuleList([nn.Linear(mlp_input_dim, hidden_dim),
                                                  nn.BatchNorm1d(hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.2),
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.BatchNorm1d(hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.2),
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.BatchNorm1d(hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Dropout(0.2),
                                                    nn.Linear(hidden_dim, input_dim)])
        # send modules to device
        self.model = self.model.to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()

        # if self.time_embed is not None:
        #     t = self.time_embed(_t)
        # else:
        t = self.time_frequencies * _t[...,None]
        t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
        context1 = self.context_frequencies * context1[...,None]
        context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
        context2 = self.context_frequencies * context2[...,None]
        context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
           # context = torch.cat((t, context), dim=-1)
        else:
            context = t

        x1 = torch.cat((_x, context1), dim=-1)
        x2 = torch.cat((_x, context2), dim=-1)

        for layer_index, layer in enumerate(self.model):
            if layer_index <= 4:
                x1 = layer(x1)
                x2 = layer(x2)
                x = weight * x1 + (1 - weight) * x2
            else:
                x = layer(x)
        
        return x

       # x = weight * x1 + (1 - weight) * x2

        #return self.model(x)

        #return self.model(_x, context1=context1, context2=context2, weight=weight)





class Conditional_ResNet_time_embed(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            freq_dim = frequencies
            self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
            self.frequencies = self.frequencies.to(device)
            self.context_dim = 2*freq_dim + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        #context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
           # context = self.frequencies * context[...,None]
           # context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context is not None:
            context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context=context)

class Conditional_ResNet_time_embed_linear_interpolation(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            freq_dim = frequencies
            self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
            self.frequencies = self.frequencies.to(device)
            self.context_dim = 2*freq_dim + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet_linear_interpolation(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        #context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
           # context = self.frequencies * context[...,None]
           # context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
            #context = 
        else:
            context = t

        return self.model(_x, context1=context1, context2=context2, weight=weight)

class Conditional_ResNet_time_embed_vector_interpolation(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            freq_dim = frequencies
            self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
            self.frequencies = self.frequencies.to(device)
            self.context_dim = 2*freq_dim + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        #context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
           # context = self.frequencies * context[...,None]
           # context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
            #context = 
        else:
            context = t
        
        return weight * self.model(_x, context=context1) + (1 - weight) * self.model(_x, context=context2)

        #return self.model(_x, context1=context1, context2=context2, weight=weight)




class Discrete_Conditional_ResNet(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, context_embed=None,
                 non_linear_context=False):
        super().__init__()

 
      #  if time_embed is None:
        freq_dim = frequencies
        self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
        self.frequencies = self.frequencies.to(device)
        if context_embed is not None:
            self.context_dim = context_embed.dim + 2*freq_dim
        else:
            self.context_dim = 2*freq_dim + context_features

        self.context_embed = context_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

       # print('context_embed', self.context_embed)

        if self.context_embed is not None:
            context = self.context_embed(context.flatten())

        t = self.frequencies * _t[...,None]
        t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)

       # print('context shape', context.shape)
       # print('t shape', t.shape)

        context = torch.cat((t, context), dim=-1)

        return self.model(_x, context=context)

class Discrete_Conditional_ResNet_linear_interpolation(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, context_embed=None):
        super().__init__()

 
      #  if time_embed is None:
        freq_dim = frequencies
        self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
        self.frequencies = self.frequencies.to(device)
        if context_embed is not None:
            self.context_dim = context_embed.dim + 2*freq_dim
        else:
            self.context_dim = 2*freq_dim + context_features

        self.context_embed = context_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet_linear_interpolation(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        if self.context_embed is not None:
            context1 = self.context_embed(context1.flatten())
            context2 = self.context_embed(context2.flatten())

        t = self.frequencies * _t[...,None]
        t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)

        context1 = torch.cat((t, context1), dim=-1)
        context2 = torch.cat((t, context2), dim=-1)

        return self.model(_x, context1=context1, context2=context2, weight=weight)

class Conditional_ResNet_linear_interpolation(nn.Module):
    def __init__(self, time_frequencies: int=3, context_frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
        else:
            self.context_dim = time_embed.dim + context_features

        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet_linear_interpolation(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context1 = self.context_frequencies * context1[...,None]
            context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
            context2 = self.context_frequencies * context2[...,None]
            context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
           # context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context1=context1, context2=context2, weight=weight)


class Conditional_ResNet_linear_interpolation_block(nn.Module):
    def __init__(self, time_frequencies: int=3, context_frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
        else:
            self.context_dim = time_embed.dim + context_features

        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet_linear_interpolation_block(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context1 = self.context_frequencies * context1[...,None]
            context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
            context2 = self.context_frequencies * context2[...,None]
            context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
           # context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context1=context1, context2=context2, weight=weight)



class Conditional_ResNet_linear_interpolation_indiv_block(nn.Module):
    def __init__(self, time_frequencies: int=3, context_frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False,
                 interp_block=2):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
        else:
            self.context_dim = time_embed.dim + context_features

        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet_linear_interpolation_indiv_block(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context,
                     interp_block=interp_block).to(device)


    def forward(self, x, context1=None, context2=None, weight=None, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()
        context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context1 = self.context_frequencies * context1[...,None]
            context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
            context2 = self.context_frequencies * context2[...,None]
            context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

            context = self.context_frequencies * context[...,None]
            context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)

            context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context1=context1, context2=context2, weight=weight, context=context)



class Conditional_ResNet_linear_interpolation_vector(nn.Module):
    def __init__(self, time_frequencies: int=3, context_frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
        else:
            self.context_dim = time_embed.dim + context_features

        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context1 = self.context_frequencies * context1[...,None]
            context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
            context2 = self.context_frequencies * context2[...,None]
            context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

        if context1 is not None:
            context1 = torch.cat((t, context1), dim=-1)
            context2 = torch.cat((t, context2), dim=-1)
           # context = torch.cat((t, context), dim=-1)
        else:
            context = t


        return weight * self.model(_x, context=context1) + (1-weight) * self.model(_x, context=context2)

class Conditional_ResNet_linear_interpolation_embedding(nn.Module):
    def __init__(self, time_frequencies: int=3, context_frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

 
        if time_embed is None:
            self.time_frequencies = torch.arange(1,time_frequencies+1,1).float()*torch.pi
            self.time_frequencies = self.time_frequencies.to(device)
            self.context_frequencies = (2**torch.arange(0,context_frequencies,1).float())*torch.pi
            self.context_frequencies = self.context_frequencies.to(device)
            #self.time_frequencies = (2**torch.arange(0,time_frequencies,1).float())*torch.pi
            #self.time_frequencies = self.time_frequencies.to(device)
            self.context_dim = 2*context_frequencies + 2*time_frequencies
        else:
            self.context_dim = time_embed.dim + context_features

        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context1=None, context2=None, weight=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        context1 = context1.flatten()
        context2 = context2.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.time_frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
            context1 = self.context_frequencies * context1[...,None]
            context1 = torch.cat((context1.cos(), context1.sin()), dim=-1).to(x.device)
            context2 = self.context_frequencies * context2[...,None]
            context2 = torch.cat((context2.cos(), context2.sin()), dim=-1).to(x.device)

            avg_context = weight * context1 + (1-weight) * context2

        if context1 is not None:
            context = torch.cat((t, avg_context), dim=-1)
            #context1 = torch.cat((t, context1), dim=-1)
            #context2 = torch.cat((t, context2), dim=-1)
           # context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context=context)
        #return weight * self.model(_x, context=context1) + (1-weight) * self.model(_x, context=context2)




import math
import torch

def sample_model_interpolation_discrete(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0, context_bins=None, max_mass=3.8, min_mass=3.2) -> Tensor:
        
        constant = (context - max_mass)/(-(max_mass - min_mass))
        context_less = torch.ones_like(context)*min_mass
        contex_high = torch.ones_like(context)*max_mass

        context_less_numpy = context_less.cpu().detach().numpy()
        contex_high_numpy = contex_high.cpu().detach().numpy()
        context_less_digitized = torch.tensor(np.digitize(context_less_numpy, context_bins)).to(context.device)
        contex_high_digitized = torch.tensor(np.digitize(contex_high_numpy, context_bins)).to(context.device)

       # context_less_digitized = torch.digitize(context_less, context_bins)

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                vt = model(input_to_model, context1=context_less_digitized, context2=contex_high_digitized, weight=constant)

            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z



class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


from torchcfm.optimal_transport import OTPlanSampler
import wandb


def train_flow(traindata, model, valdata=None, optimizer=None, 
               num_epochs=2, batch_size=5,
               device=torch.device('cuda:0'),sigma_fm=0.001, 
               save_model = False, model_path=None, interval=200,
    ot=False, compute_log_likelihood=False,
    likelihood_interval=20, likelihood_start=300,
    scheduler=False,
    early_stop_patience=50,
    wandb_log=False):

    if compute_log_likelihood:
        print(f'valdata is {valdata.shape}')
    if ot:
        ot_sampler = OTPlanSampler(method="exact")
    start = time.time()
    losses = []
    logprob_list = []
    logprob_epoch = []
    early_stop_counter = 0

    best_val_log_prob=10000000
    for epoch in range(num_epochs):
        data_ = create_loader(traindata, shuffle=True)
        running_loss = 0
        if epoch % 1 == 0:
            print('epoch', epoch)

        for i in range(len(data_)//batch_size+1):
            optimizer.zero_grad()
            
            x1 = data_[i*batch_size:(i+1)*batch_size].to(device)

            context = x1[:,0].reshape(-1,1)
            x1 = x1[:,1:-1]

            x0 = torch.randn_like(x1).to(device)
            t = torch.rand_like(x0[:, 0].reshape(-1,1)).to(device)

            xt = sample_conditional_pt(x0, x1, t, sigma_fm)
            if ot:
                pi = ot_sampler.get_map(x0, x1)
                i, j = ot_sampler.sample_map(pi, x0.shape[0], replace=False)
                xt = x1[j]
                t = t[j]
                x0 = x0[i]
            

            ut = compute_conditional_vector_field(x0, x1, sigma_fm)

            vt = model(torch.cat([xt, t], dim=-1),context=context)
            loss = torch.mean((vt - ut) ** 2)

            running_loss += loss.item()
           
            loss.backward()
            optimizer.step()

        
        total_loss = running_loss/(len(data_)//batch_size+1)
        losses.append(total_loss)
    
        # if total_loss is lower than all losses in best_loss_array, save model
        # if save_model:
        #     if num_epochs-epoch < 11: 
        #         print('saving model at epoch', epoch)
        #         torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')

        if compute_log_likelihood:
            if (likelihood_start < epoch+1) and (epoch % likelihood_interval == 0): 
                log_prob_ = compute_log_prob(model, valdata[:,1:-1], 
                                    valdata[:,0].reshape(-1,1), 
                                    batch_size=50_000,
                                    device=device,
                                    method='torchdyn')
                mean_log_prob = np.mean(-log_prob_)
                if wandb_log:
                    wandb.log({'logprob': mean_log_prob, 'lr': optimizer.param_groups[0]['lr']})
                if mean_log_prob < best_val_log_prob:
                    best_val_log_prob = mean_log_prob
                    early_stop_counter = 0
                else:
                    early_stop_counter +=1

                    #torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')
                if early_stop_counter > early_stop_patience:
                    print('Early stopping at epoch', epoch)
                    break


                if scheduler:
                    scheduler.step(mean_log_prob)
                logprob_list.append(mean_log_prob)
                logprob_epoch.append(epoch)
                torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')

                print(f'saving logprob for epoch {epoch}')

    end = time.time()
    print('Time taken: ', end - start)
    if not compute_log_likelihood:
        return losses
    else:
        return losses, logprob_list, logprob_epoch

def sample(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0) -> Tensor:


        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
               # context = data[:,-1].reshape(-1,1)
               # print
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                vt = model(input_to_model, context=context)

            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z

def sample_minibatches(model: torch.nn.Module , x: Tensor, context: Tensor,
                        start:float=0.0, end:int=1.0, batch_size=10_000):
    samples = []
    for i in range(len(x)//batch_size + 1):
        print(f'Sampling batch {i}')
        test_input = x[i*batch_size:(i+1)*batch_size].to(x.device)
        test_context = context[i*batch_size:(i+1)*batch_size].to(x.device)
        #test_data = torch.cat([test_context,test_input], dim=-1)
        samples.append(sample(model, test_input, test_context, start, end))
    
    #samples = torch.tensor(samples)
    samples = torch.cat(samples, dim=0)

    return samples


def compute_log_prob(model, data, context, batch_size=50_000,
                     device=torch.device('cuda:0'),
                     method='zuko'):
    log_likelihood = []
    model.eval()
    for i in range(len(data)//batch_size + 1):
        print(f'Computing log likelihood for batch {i}')
        test_input = data[i*batch_size:(i+1)*batch_size].to(device)
        test_context = context[i*batch_size:(i+1)*batch_size].to(device)
        test_data = torch.cat([test_context,test_input], dim=-1)
       # print(test_input.shape)
        if method=='zuko':
            z, log_likelihood_ = log_prob(model, test_input, 
                                          test_context, start=1.0,end=0.0)
        elif method=='torchdyn':
            _, log_likelihood_ = log_prob_torchdyn(model, test_data,
                                                   device=device)
        
        log_likelihood.append(log_likelihood_.detach().cpu().numpy())

        # print(test_input.shape)
        #log_likelihood.append(log_prob(model, test_input,start=1,end=0).detach().cpu().numpy())

    log_likelihood = np.concatenate(log_likelihood)

    return log_likelihood


def sample_model_interpolation(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0, max_mass=3.8, min_mass=3.2) -> Tensor:
        
        constant = (context - max_mass)/(-(max_mass-min_mass))
        context_less = torch.ones_like(context)*min_mass
        contex_high = torch.ones_like(context)*max_mass

        #context_less_numpy = context_less.cpu().detach().numpy()
        #contex_high_numpy = contex_high.cpu().detach().numpy()
        #context_less_digitized = torch.tensor(np.digitize(context_less_numpy, context_bins)).to(context.device)
        #contex_high_digitized = torch.tensor(np.digitize(contex_high_numpy, context_bins)).to(context.device)

       # context_less_digitized = torch.digitize(context_less, context_bins)

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                vt = model(input_to_model, context1=context_less, context2=contex_high, weight=constant)


            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z


def sample_model_interpolation_block(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0, max_mass=3.8, min_mass=3.2) -> Tensor:
        
        constant = (context - max_mass)/(-(max_mass-min_mass))
        context_less = torch.ones_like(context)*min_mass
        contex_high = torch.ones_like(context)*max_mass

        #context_less_numpy = context_less.cpu().detach().numpy()
        #contex_high_numpy = contex_high.cpu().detach().numpy()
        #context_less_digitized = torch.tensor(np.digitize(context_less_numpy, context_bins)).to(context.device)
        #contex_high_digitized = torch.tensor(np.digitize(contex_high_numpy, context_bins)).to(context.device)

       # context_less_digitized = torch.digitize(context_less, context_bins)

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                vt = model(input_to_model, 
                           context1=context_less, 
                           context2=contex_high, 
                           weight=constant,
                           context=context)


            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z
from src.generate_data_lhc import *
