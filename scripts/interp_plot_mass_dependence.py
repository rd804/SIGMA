import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('.')
import os
#nsig=1000

#CR_path = f'./results/debugging_nflow_interp/non_linear_activation/base_data'
save_dir=f'./results/debugging_nflow_interp/figures'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


nsig=1000
savedir_CR = f'./results/debugging_nflow_interp/non_linear_activation/base_CR'
#savedir_SR = f'./results/debugging_nflow_interp/non_linear_activation/base_SR/'
savedir_data = f'./results/debugging_nflow_interp/non_linear_activation/base_data'
#savdir_no_signal = f'./results/debugging_nflow_interp/non_linear_activation/base_no_signal/'

import torch
from torch import Tensor
from zuko.utils import odeint

def sample_model_interpolation(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0) -> Tensor:
        
        constant = (context - 3.8)/(-0.6)
        context_less = torch.ones_like(context)*3.2
        contex_high = torch.ones_like(context)*3.8

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
               # context = data[:,-1].reshape(-1,1)
               # print
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
               # vt_less = model(input_to_model, context=context_less)
               # vt_high = model(input_to_model, context=contex_high)
                vt = model(input_to_model, context1=context_less, context2=contex_high, weight=constant)
                #vt = 0.5*(vt_less + vt_high)
                #vt = constant * vt_less + (1-constant) * vt_high
              #  print('vt shape', vt.shape)
              #  print('v_t less shape', vt_less.shape)

            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z

from src.generate_data_lhc import *
data_dir='data/extended1'
SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = 1000, resample_seed = 1,resample = False)

SR_data = np.concatenate([SR_data[:,:5], SR_data[:,-1].reshape(-1,1)], axis=1)
CR_data = np.concatenate([CR_data[:,:5], CR_data[:,-1].reshape(-1,1)], axis=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from src.flow_matching import *
data = torch.from_numpy(SR_data[:2000,1:-1]).to(device).float()
torch.manual_seed(0)
noise1 = torch.randn_like(data).to(device).float()
mass = torch.from_numpy(SR_data[:2000,0].reshape(-1,1)).to(device).float()


t=0.2

constant = (mass - 3.8)/(-0.6)
constant_numpy = constant.detach().cpu().numpy()
context_less_1 = torch.ones_like(mass)*3.2
context_high_1 = torch.ones_like(mass)*3.8
context_less_2 = torch.ones_like(mass)*3.3
context_high_2 = torch.ones_like(mass)*3.7
context_less_0 = torch.ones_like(mass)*3.0
context_high_0 = torch.ones_like(mass)*4.0

model = Conditional_ResNet(frequencies=3, 
                                   context_features=1, 
                                input_dim=4, device=device,
                                hidden_dim=256, num_blocks=4, 
                                use_batch_norm=True, 
                                dropout_probability=0.2)

interp_model = Conditional_ResNet_linear_interpolation(frequencies=3,
                                     context_features=1, 
                                 input_dim=4, device=device,
                                 hidden_dim=256, num_blocks=4, 
                                 use_batch_norm=True, 
                                 dropout_probability=0.2)
i=78
full_list = []
full_list_CR = []
full_mass_list = []
full_mass_list_CR = []

interp_linear_list = []
interp_model_list = []

interp_linear_list_CR = []
interp_model_list_CR = []

for try_ in [0]:
    filelist = os.listdir(f'{savedir_data}_{try_}')
    files_CR = os.listdir(f'{savedir_CR}_{try_}')

    print(f'{savedir_data}_{try_}')
    print(f'{savedir_CR}_{try_}')

    filelist = [f for f in filelist if 'model' in f]

    filelist_CR = [f for f in files_CR if 'model' in f]



    for file in [filelist[0]]:
        model.eval()
        interp_model.eval()
        model.load_state_dict(torch.load(f'{savedir_data}_{try_}/{file}'))
        interp_model.load_state_dict(torch.load(f'{savedir_data}_{try_}/{file}'))
        with torch.no_grad():
            t_array = torch.ones(data.shape[0], 1).to(data.device) * t
            input_to_model = torch.cat([data,t_array], dim=-1)
            vt = model(input_to_model, context=mass).detach().cpu().numpy()
            vt_less_1 = model(input_to_model, context=context_less_1).detach().cpu().numpy()
            vt_high_1 = model(input_to_model, context=context_high_1).detach().cpu().numpy()
            vt_less_2 = model(input_to_model, context=context_less_2).detach().cpu().numpy()
            vt_high_2 = model(input_to_model, context=context_high_2).detach().cpu().numpy()
            vt_less_0 = model(input_to_model, context=context_less_0).detach().cpu().numpy()
            vt_high_0 = model(input_to_model, context=context_high_0).detach().cpu().numpy()
            vt_interp = constant_numpy * vt_less_1 + (1-constant_numpy) * vt_high_1
            vt_interp_model = interp_model(input_to_model, context1=context_less_1, context2=context_high_1, weight=constant).detach().cpu().numpy()

        vt_array = [vt_less_0[i,:],vt_high_0[i,:],vt_less_1[i,:], vt_high_1[i,:], vt_less_2[i,:], vt_high_2[i,:], vt[i,:]]
        context_array = [3.0,4.0,3.2, 3.8, 3.3, 3.7, mass[i].detach().cpu().numpy()[0]]

        full_list.extend(vt_array)
        full_mass_list.extend(context_array)
        interp_linear_list.extend(vt_interp[i,:])
        interp_model_list.extend(vt_interp_model[i,:])



    model = Conditional_ResNet(frequencies=3, 
                                    context_features=1, 
                                    input_dim=4, device=device,
                                    hidden_dim=256, num_blocks=4, 
                                    use_batch_norm=True, 
                                    dropout_probability=0.2)

    interp_model = Conditional_ResNet_linear_interpolation(frequencies=3,
                                        context_features=1, 
                                        input_dim=4, device=device,
                                        hidden_dim=256, num_blocks=4, 
                                        use_batch_norm=True, 
                                        dropout_probability=0.2)


    for file in [filelist_CR[0]]:
        model.eval()
        interp_model.eval()
        model.load_state_dict(torch.load(f'{savedir_CR}_{try_}/{file}'))
        interp_model.load_state_dict(torch.load(f'{savedir_CR}_{try_}/{file}'))
        with torch.no_grad():
            t_array = torch.ones(data.shape[0], 1).to(data.device) * t
            input_to_model = torch.cat([data,t_array], dim=-1)
            vt_CR = model(input_to_model, context=mass).detach().cpu().numpy()
            vt_CR_less_1 = model(input_to_model, context=context_less_1).detach().cpu().numpy()
            vt_CR_high_1 = model(input_to_model, context=context_high_1).detach().cpu().numpy()
            vt_CR_less_2 = model(input_to_model, context=context_less_2).detach().cpu().numpy()
            vt_CR_high_2 = model(input_to_model, context=context_high_2).detach().cpu().numpy()
            vt_CR_less_0 = model(input_to_model, context=context_less_0).detach().cpu().numpy()
            vt_CR_high_0 = model(input_to_model, context=context_high_0).detach().cpu().numpy()
            vt_CR_interp = constant_numpy * vt_CR_less_1 + (1-constant_numpy) * vt_CR_high_1
            vt_CR_interp_model = interp_model(input_to_model, context1=context_less_1, context2=context_high_1, weight=constant).detach().cpu().numpy()

        vt_CR_array = [vt_CR_less_0[i,:],vt_CR_high_0[i,:],vt_CR_less_1[i,:], vt_CR_high_1[i,:], vt_CR_less_2[i,:], vt_CR_high_2[i,:], vt_CR[i,:]]
        context_array = [3.0,4.0,3.2, 3.8, 3.3, 3.7, mass[i].detach().cpu().numpy()[0]]

        full_list_CR.extend(vt_CR_array)
        full_mass_list_CR.extend(context_array)
        interp_linear_list_CR.extend(vt_CR_interp[i,:])
        interp_model_list_CR.extend(vt_CR_interp_model[i,:])
#i_list =[40]

print('length full list', len(full_list))
print('length full list CR', len(full_list_CR))
print('length full mass list', len(full_mass_list))
print('length full mass list CR', len(full_mass_list_CR))
print('length interp linear list', len(interp_linear_list))
print('length interp model list', len(interp_model_list))
print('length interp linear list CR', len(interp_linear_list_CR))
print('length interp model list CR', len(interp_model_list_CR))

print(interp_linear_list)

for feature in range(4):
    plt.scatter(full_mass_list, [k[feature] for k in full_list], label='data')
    plt.scatter(full_mass_list_CR, [k[feature] for k in full_list_CR], label='CR')
    plt.scatter(full_mass_list[-1], interp_linear_list[feature], label='linear interp (data)')
    plt.scatter(full_mass_list[-1], interp_model_list[feature] , label='context interp (data)')
    plt.scatter(full_mass_list_CR[-1], interp_linear_list_CR[feature], label='linear interp (CR)')
    plt.scatter(full_mass_list_CR[-1], interp_model_list_CR[feature], label='context interp (CR)')
    plt.title(f'vt {feature}')
    plt.legend()
    plt.savefig(f'{save_dir}/vt_{feature}.png')
    plt.close()
#    full_list = []
#    full_list_CR = []
#    full_mass_list = []
#    for i in i_list:
#       vt_array = [vt_less_0[i,feature],vt_high_0[i,feature],vt_less_1[i,feature], vt_high_1[i,feature], vt_less_2[i,feature], vt_high_2[i,feature], vt[i,feature]]
#       vt_CR_array = [vt_CR_less_0[i,feature],vt_CR_high_0[i,feature],vt_CR_less_1[i,feature], vt_CR_high_1[i,feature], vt_CR_less_2[i,feature], vt_CR_high_2[i,feature], vt_CR[i,feature]]
#       context_array = [3.0,4.0,3.2, 3.8, 3.3, 3.7, mass[i].detach().cpu().numpy()[0]]

#     #  data_interp = [vt_interp[i,feature]]
#     #  CR_interp = [vt_CR_interp[i,feature]]

#     #  data_interp_model = [vt_interp_model[i,feature]]
#     #  CR_interp_model = [vt_CR_interp_model[i,feature]]

#       full_list.extend(vt_array)
#       full_mass_list.extend(context_array)
#       full_list_CR.extend(vt_CR_array)

#    plt.scatter(full_mass_list, full_list, label='data')
#    plt.scatter(full_mass_list, full_list_CR, label='CR')
  # plt.scatter(full_mass_list[-1], data_interp, label='data interp')
  # plt.scatter(full_mass_list[-1], CR_interp, label='CR interp')
  # plt.scatter(full_mass_list[-1], data_interp_model, label='data interp model')
  # plt.scatter(full_mass_list[-1], CR_interp_model, label='CR interp model')
#    plt.title(f'vt {feature}')
#    plt.legend()
#    plt.savefig(f'{save_dir}/vt_{feature}.png')
#    plt.show()
   # plt.scatter(context_array, vt_array)
   # plt.scatter(context_array[-1], vt_array[-1])
   # plt.show()
