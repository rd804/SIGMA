import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('.')
import os
from src.flow_matching import *
from src.utils import *
from sklearn.utils.class_weight import compute_sample_weight

n_sig=1000

#CR_path = f'./results/debugging_nflow_interp/non_linear_activation/base_data'
save_dir=f'./results/debugging_nflow_interp/figures'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


model_dir=f'./results/debugging_nflow_interp/context_embed_{n_sig}/base_dR_data'
n_sig=1000

#nsig=1000
#savedir_CR = f'./results/debugging_nflow_interp/nonlin_embed_2000/base_dR_CR'
#savedir_SR = f'./results/debugging_nflow_interp/non_linear_activation/base_SR/'
savedir_data = f'./results/debugging_nflow_interp/context_embed_{n_sig}/base_dR_data'
savedir_CR = f'./results/debugging_nflow_interp/context_embed_{n_sig}/base_dR_CR'
savedir_SR = f'./results/debugging_nflow_interp/context_embed_{n_sig}/base_dR_SR'

#savdir_no_signal = f'./results/debugging_nflow_interp/non_linear_activation/base_no_signal/'
sic_CR = np.load(savedir_CR + f'/sic_cathode.npy')
tpr_CR = np.load(savedir_CR + f'/tpr_cathode.npy')

sic_SR = np.load(savedir_SR + f'/sic_cathode.npy')
tpr_SR = np.load(savedir_SR + f'/tpr_cathode.npy')

sic_data = np.load(savedir_data + f'/sic_cathode.npy')
tpr_data = np.load(savedir_data + f'/tpr_cathode.npy')

sic_iad = np.load(savedir_data + f'/sic_iad.npy')
tpr_iad = np.load(savedir_data + f'/tpr_iad.npy')


context_bins = np.load(f'{model_dir}/context_bins.npy')

import torch
from torch import Tensor
from zuko.utils import odeint


def sample_model_interpolation(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0, context_bins=context_bins) -> Tensor:
        
        constant = (context - 3.8)/(-0.6)
        context_less = torch.ones_like(context)*3.2
        contex_high = torch.ones_like(context)*3.8

        context_less_numpy = context_less.cpu().detach().numpy()
        contex_high_numpy = contex_high.cpu().detach().numpy()
        context_less_digitized = torch.tensor(np.digitize(context_less_numpy, context_bins)).to(context.device)
        contex_high_digitized = torch.tensor(np.digitize(contex_high_numpy, context_bins)).to(context.device)

       # context_less_digitized = torch.digitize(context_less, context_bins)

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
               # context = data[:,-1].reshape(-1,1)
               # print
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
               # vt_less = model(input_to_model, context=context_less)
               # vt_high = model(input_to_model, context=contex_high)
                vt = model(input_to_model, context1=context_less_digitized, context2=contex_high_digitized, weight=constant)
                #vt = 0.5*(vt_less + vt_high)
                #vt = constant * vt_less + (1-constant) * vt_high
              #  print('vt shape', vt.shape)
              #  print('v_t less shape', vt_less.shape)

            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z

from src.generate_data_lhc import *
data_dir='data/baseline_delta_R'
#data_dir='data/extended1'
SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = n_sig, resample_seed = 1,resample = False)
#SR_data = np.concatenate([SR_data[:,:5], SR_data[:,-1].reshape(-1,1)], axis=1)
#CR_data = np.concatenate([CR_data[:,:5], CR_data[:,-1].reshape(-1,1)], axis=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mass = torch.from_numpy(SR_data[:,0].reshape(-1,1)).to(device).float()
data = torch.from_numpy(SR_data[:,1:-1]).to(device).float()
noise1 = torch.randn_like(data).to(device).float()
noise2 = torch.randn_like(data).to(device).float()
noise3 = torch.randn_like(data).to(device).float()
noise = torch.cat([noise1, noise2], dim=0)
noise = torch.cat([noise, noise3], dim=0)
mass = torch.cat([mass, mass], dim=0)
mass = torch.cat([mass, mass,mass], dim=0)

mini_batch_length = len(noise)//10


import pickle
with open(f'{model_dir}/pre_parameters.pkl', 'rb') as f:
    pre_parameters_cpu = pickle.load(f)

pre_parameters = {}
for key in pre_parameters_cpu.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters_cpu[key].astype('float32')).to(device)


_x_test = np.load(f'{data_dir}/x_test.npy')
#_x_test = np.concatenate([_x_test[:,:5], _x_test[:,-1].reshape(-1,1)], axis=1)
x_test = preprocess_params_transform(_x_test, pre_parameters_cpu)
x_SR = preprocess_params_transform(SR_data, pre_parameters_cpu)

val_logprob = np.load(f'{model_dir}/val_logprob.npy')
val_logprob_epoch = np.load(f'{model_dir}/val_logprob_epoch.npy')

lowest10 = np.argsort(val_logprob)[:10]
lowest_epochs = val_logprob_epoch[lowest10]


from sklearn.ensemble import HistGradientBoostingClassifier
#filelist = os.listdir(f'{model_dir}')
#filelist = [f for f in filelist if 'model' in f]


ensembled_samples = []
ensembled_mass = []
interp_model = Discrete_Conditional_ResNet_linear_interpolation(frequencies=3,
                                     context_features=1, 
                                 input_dim=5, device=device,
                                 hidden_dim=256, num_blocks=4, 
                                 use_batch_norm=True, 
                                 dropout_probability=0.2,
                                 context_embed=SinusoidalPosEmb(dim=32,theta=100))

for i,file in enumerate(lowest_epochs):
   print(file)
   interp_model.load_state_dict(torch.load(f'{model_dir}/model_epoch_{file}.pth'))
   #model.load_state_dict(torch.load(f'{CR_path}model_epoch_{file}.pth'))
   noise_batch = noise[i*mini_batch_length:(i+1)*mini_batch_length]
   mass_batch = mass[i*mini_batch_length:(i+1)*mini_batch_length] 
   samples = sample_model_interpolation(interp_model, noise_batch, mass_batch, start=0.0, end=1.0)
   #samples = sample_interpolation_(model, noise_batch, mass_batch, start=0.0, end=1.0)
   ensembled_samples.append(samples)
   ensembled_mass.append(mass_batch)

mass = torch.cat(ensembled_mass, dim=0)
samples = torch.cat(ensembled_samples, dim=0)
samples = torch.concat([mass, samples], axis=1)
samples = torch.concat([samples, torch.ones(samples.shape[0],1).to(device)], axis=1)

samples_inverse = inverse_transform(samples, pre_parameters)

n_features = samples_inverse.shape[1]-1

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
    plt.savefig(f'{savedir_data}/feature_interpolation_{i}.png')
    plt.close()

print('samples saved')

predict_cathode = []
cathode_data = np.vstack((x_SR[:,1:-1], samples.detach().cpu().numpy()[:,1:-1]))
cathode_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(samples.shape[0])])
sample_weights_cathode = compute_sample_weight(class_weight='balanced', y=cathode_labels)

for seed in range(50):
   print(f'train classifier {seed}')
   ###################################
   # Train classifier on cathode data #

   clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
   clf.fit(cathode_data, cathode_labels,sample_weight=sample_weights_cathode)
   predict_cathode.append(clf.predict_proba(x_test[:,1:-1])[:,1])

predict_cathode = np.mean(predict_cathode, axis=0)
sic_score_cathode , tpr_score_cathode , _ = SIC_cut(x_test[:,-1], predict_cathode)


plt.plot(tpr_score_cathode, sic_score_cathode, label='interp (data')
#plt.plot(tpr_no_signal, sic_no_signal, label='no signal')
plt.plot(tpr_SR, sic_SR, label='SR')
plt.plot(tpr_CR, sic_CR, label='CR')
plt.plot(tpr_data, sic_data, label='data')
plt.plot(tpr_iad, sic_iad, label='IAD', color='black')
plt.title(rf'Nsig={n_sig}, baseline + $\Delta$R')
plt.legend()
plt.savefig(f'{savedir_data}/sic_tpr_cathode_interp.png')
plt.close()

