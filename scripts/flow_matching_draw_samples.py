# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
#print(sys.path)
sys.path.append('.')
#from nflows.nn.nets import ResidualNet
#from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
#from src.flows import *
from src.flow_matching import *
import os
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
# import train_test_split
from sklearn.model_selection import train_test_split, ShuffleSplit
import argparse
import wandb
import pickle
import sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
#os.environ["CUDA_VISIBLE_DEVICES"]='2'
from scipy.stats import rv_histogram



#device = torch.device(f'cuda:{local_rank}' if CUDA else "cpu")   
device = torch.device('cuda:0')
#data_dir = 'data/baseline_delta_R'
#model_path = 'results/debugging_nflow_interp/nonlin_embed_2000/base_dR_data'
data_dir='data/mx_100_my_500'
model_path='results//mx_100_my_500/nsig_1000/seed_0/'


SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = 1000, resample_seed = 0,resample = False)


print('x_train shape', CR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

#baseline = CR_data[:,:5]
#CR_data = np.concatenate([baseline, CR_data[:,-1].reshape(-1,1)], axis=1)
n_features = int(CR_data.shape[1]-2)
print('n_features', n_features)

with open(f'{model_path}/pre_parameters.pkl','rb') as f:
    pre_parameters = pickle.load(f)

SR_preprocessed = preprocess_params_transform(SR_data, pre_parameters)
CR_preprocessed = preprocess_params_transform(CR_data, pre_parameters)

_x_test = np.load(f'{data_dir}/x_test.npy')
x_test = preprocess_params_transform(_x_test, pre_parameters)
x_SR = preprocess_params_transform(SR_data, pre_parameters)


# model = Conditional_ResNet(frequencies=3, 
#                             context_features=1, 
#                         input_dim=n_features, device=device,
#                         hidden_dim=256, num_blocks=4, 
#                         use_batch_norm=True, 
#                         dropout_probability=0.2,
#                         non_linear_context=True)

model = Conditional_ResNet_time_embed(frequencies=3, 
                                context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=256, num_blocks=4, 
                                use_batch_norm=True, 
                                dropout_probability=0.2,
                                non_linear_context=False)

SR_mass = SR_data[:,0]
SR_hist = np.histogram(SR_mass, bins=60, density=True)
SR_density = rv_histogram(SR_hist)

noise = torch.randn(2_000_000, n_features).to(device).float()
mass_samples_SR = SR_density.rvs(size=len(noise))
mass_samples_SR = mass_samples_SR.reshape(-1,1)
mass = torch.from_numpy(mass_samples_SR).to(device).float()


# mass = torch.from_numpy(SR_data[:,0].reshape(-1,1)).to(device).float()
# data = torch.from_numpy(SR_data[:,1:-1]).to(device).float()
# noise1 = torch.randn_like(data).to(device).float()
# noise2 = torch.randn_like(data).to(device).float()
# noise3 = torch.randn_like(data).to(device).float()
# noise4 = torch.randn_like(data).to(device).float()
# noise5 = torch.randn_like(data).to(device).float()

# noise = torch.cat([noise1, noise2], dim=0)
# noise = torch.cat([noise, noise3], dim=0)
# mass = torch.cat([mass, mass, mass], dim=0)

mini_batch_length = len(noise)//10

ensembled_samples = []
ensembled_mass = []

log_prob_mean = np.load(f'{model_path}/val_logprob.npy')
logprob_epoch = np.load(f'{model_path}/val_logprob_epoch.npy')

log_prob_mean_sorted = np.argsort(log_prob_mean)
logprob_epoch = np.array(logprob_epoch)
lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]


for i,epoch in enumerate(lowest_epochs):
    model.load_state_dict(torch.load(f'{model_path}/model_epoch_{epoch}.pth'))
    noise_batch = noise[i*mini_batch_length:(i+1)*mini_batch_length]
    mass_batch = mass[i*mini_batch_length:(i+1)*mini_batch_length] 
    samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
    ensembled_samples.append(samples)
    ensembled_mass.append(mass_batch)

mass = torch.cat(ensembled_mass, dim=0)
samples = torch.cat(ensembled_samples, dim=0)
# print(samples.shape)
# print(mass.shape)
samples = torch.concat([mass, samples], axis=1)
samples = torch.concat([samples, torch.ones(samples.shape[0],1).to(device)], axis=1)

pre_parameters_gpu = {key: torch.tensor(value).to(device) for key, value in pre_parameters.items()}

samples_inverse = inverse_transform(samples, pre_parameters_gpu)
print(samples_inverse.shape)

np.save(f'{model_path}/2M_samples.npy', samples_inverse.cpu().detach().numpy())

for i in range(1,n_features+1,1):
    figure = plt.figure()
    #bins = np.arange(0, 1, 0.03)
    max_ = np.max(SR_data[:,i])
    min_ = np.min(SR_data[:,i])
    bins = np.linspace(min_, max_, num=50)
    plt.hist(SR_data[:,i],bins=bins,density=True, histtype='stepfilled', label='data', color='gray', alpha=0.5)
    plt.hist(samples_inverse[:,i].cpu().detach().numpy(),bins=bins,density=True,
            histtype='step', label='samples')
    plt.hist(_x_test[:,i][_x_test[:,-1]==0], bins=bins, density=True, histtype='step', label='true background', color='black')
    plt.legend()
    plt.savefig(f'{model_path}/feature_{i}.png')
    plt.close()



