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
from scipy.stats import rv_histogram

#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--time_frequencies', type=int, default=3)
parser.add_argument('--context_frequencies', type=int, default=10)
parser.add_argument('--x_train', type=str, default='CR')
parser.add_argument('--ensemble_size', type=int, default=50)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--data_dir', type=str, default='data/extended1', help='data directory')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--baseline', action='store_true', help='if baseline is used')

parser.add_argument('--context_embedding', action='store_true', help='if time embedding is used')
parser.add_argument('--non_linear_context', action='store_true', help='if non linear context is used')
parser.add_argument('--scaled_mass', action='store_true', help='if mass is scaled')
parser.add_argument('--sample_interpolated', action='store_true', help='if interpolated samples are generated')
parser.add_argument('--higher_mass', default=3.8, type=float, help='higher mass')
parser.add_argument('--lower_mass', default=3.2, type=float, help='lower mass')
#parser.add_argument('--try', type=int, default=0)
parser.add_argument('--interpolation_method', type=str, default='linear')
parser.add_argument('--interp_block', type=int, default=1)

parser.add_argument('--linear_interpolation', action='store_true', help='if linear interpolation is used')


parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_flow_matching')
parser.add_argument('--wandb_job_type', type=str, default='test_time_embedding')
parser.add_argument('--wandb_run_name', type=str, default='extended3')


# rank          = int(os.environ["SLURM_PROCID"])
# gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
# local_rank = rank - gpus_per_node * (rank // gpus_per_node)



args = parser.parse_args()
#args.interp_block = [2,3,4]
#args.wandb_run_name = f'{args.wandb_run_name}_{local_rank}'

save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'

# if os.path.exists(f'{save_path}predict_cathode.npy'):
#     print(f'already done {args.wandb_run_name}')
#     sys.exit()

#save_path = f'{save_path}_'

# if os.path.exists(f'{save_path}best_val_loss_scores.npy'):
#     print(f'already done {args.wandb_run_name}')
#     sys.exit()


if not os.path.exists(save_path):
    os.makedirs(save_path)

CUDA = True

#device = torch.device(f'cuda:{local_rank}' if CUDA else "cpu")   
device = torch.device(args.device if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="fast_interpolation", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)

SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

if args.baseline:
    SR_data = np.concatenate([SR_data[:,:5], SR_data[:,-1].reshape(-1,1)], axis=1)
    CR_data = np.concatenate([CR_data[:,:5], CR_data[:,-1].reshape(-1,1)], axis=1)

#n_features = 4

print('x_train shape', CR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

#baseline = CR_data[:,:5]
#CR_data = np.concatenate([baseline, CR_data[:,-1].reshape(-1,1)], axis=1)
n_features = int(CR_data.shape[1]-2)
print('n_features', n_features)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma, 'n_features':n_features})

if args.x_train == 'CR':
    pre_parameters_cpu = preprocess_params_fit(CR_data)
    x_train = preprocess_params_transform(CR_data, pre_parameters_cpu)
    print('training on CR, training data shape', x_train.shape)
elif args.x_train == 'SR':
    pre_parameters_cpu = preprocess_params_fit(SR_data)
    x_train = preprocess_params_transform(SR_data, pre_parameters_cpu)
    print('training on SR, training data shape', x_train.shape)
elif args.x_train == 'data':
    data_ = np.vstack((CR_data, SR_data))
    data_ = shuffle(data_)
    print('training on data, training data shape', data_.shape)
    pre_parameters_cpu = preprocess_params_fit(data_)
    x_train = preprocess_params_transform(data_, pre_parameters_cpu)
elif args.x_train == 'no_signal':
    data_ = np.vstack((CR_data, SR_data))
    data_ = shuffle(data_)
    data_ = data_[data_[:,-1]==0]
    print('training on no signal, training data shape', data_.shape)
    pre_parameters_cpu = preprocess_params_fit(data_)
    x_train = preprocess_params_transform(data_, pre_parameters_cpu)

# save pre_parameters
# with open(save_path+'pre_parameters.pkl', 'wb') as f:
#     pickle.dump(pre_parameters_cpu, f)
with open(f'{save_path}pre_parameters.pkl', 'rb') as f:
    pre_parameters_cpu = pickle.load(f)

if args.context_embedding:
    max_mass = np.max(x_train[:,0])
    min_mass = np.min(x_train[:,0])
    context_bins = np.arange(min_mass, max_mass, 0.1)

    np.save(f'{save_path}context_bins.npy', context_bins)

    x_train[:,0] = np.digitize(x_train[:,0], context_bins)

if args.scaled_mass:
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(-1,1))
    #x_train[:,0] = scaler.fit_transform(x_train[:,0].reshape(-1,1)).reshape(-1)

    # save scaler
    #with open(save_path+'scaler.pkl', 'wb') as f:
     #   pickle.dump(scaler, f)
    with open(f'{save_path}scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)



if not args.shuffle_split: 
    data_train, data_val = train_test_split(x_train, test_size=0.15, random_state=args.seed)

    #data_train = x_train   
    #data_train, data_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.15, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break



_x_test = np.load(f'{args.data_dir}/x_test.npy')
if args.baseline:
    _x_test = np.concatenate([_x_test[:,:5], _x_test[:,-1].reshape(-1,1)], axis=1)
x_test = preprocess_params_transform(_x_test, pre_parameters_cpu)
x_SR = preprocess_params_transform(SR_data, pre_parameters_cpu)

if args.scaled_mass:
    x_test[:,0] = scaler.transform(x_test[:,0].reshape(-1,1)).reshape(-1)
    x_SR[:,0] = scaler.transform(x_SR[:,0].reshape(-1,1)).reshape(-1)

if args.context_embedding:
    x_train[:,0] = np.digitize(x_train[:,0], context_bins)
    x_test[:,0] = np.digitize(x_test[:,0], context_bins)
    x_SR[:,0] = np.digitize(x_SR[:,0], context_bins)




traintensor = torch.from_numpy(data_train.astype('float32')).to(device)
valtensor = torch.from_numpy(data_val.astype('float32')).to(device)
testtensor = torch.from_numpy(x_test.astype('float32')).to(device)

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
print('X_test shape', testtensor.shape)

pre_parameters = {}
for key in pre_parameters_cpu.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters_cpu[key].astype('float32')).to(device)


for i in range(1):
    print(f'Ensemble {i}')

    if args.context_embedding:
        print('context embedding sinusoidal')
        model = Discrete_Conditional_ResNet(context_features=1,
                            frequencies=args.time_frequencies, 
                            input_dim=n_features, 
                            device=device, 
                            hidden_dim=args.hidden_dim, 
                            num_blocks=args.num_blocks,
                            use_batch_norm=True, 
                            dropout_probability=0.2, 
                            context_embed=SinusoidalPosEmb(dim=args.context_frequencies,theta=100),
                            non_linear_context=args.non_linear_context)
       # print(model)
    else:
        if not args.x_train == 'CR':
            # print('no context embedding')
            print('Model with continuous mass embedding')
            model = Conditional_ResNet(context_frequencies=args.context_frequencies,
                                    time_frequencies=args.time_frequencies, 
                                    context_features=1, 
                                        input_dim=n_features, device=device,
                                        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                        use_batch_norm=True, 
                                        dropout_probability=0.2,
                                        non_linear_context=args.non_linear_context)
            
        elif args.x_train == 'CR':
            print('Since we are training CR, model without continuous mass embedding is used')
            model = Conditional_ResNet_time_embed(frequencies=args.time_frequencies, 
                                context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                use_batch_norm=True, 
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)



                            
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
    # trainloss, logprob_list, logprob_epoch = train_flow(traintensor, 
    #         model, valdata=valtensor ,optimizer=optimizer,
    #         num_epochs=args.epochs, batch_size=args.batch_size,
    #         device=device, sigma_fm=0.001,
    #         save_model=True, model_path=f'{save_path}',
    #         compute_log_likelihood=True,
    #         likelihood_interval=5, likelihood_start=100,
    #         early_stop_patience=20,
    #         wandb_log=args.wandb,
    #         scheduler=scheduler)

logprob_list = np.load(f'{save_path}val_logprob.npy')
logprob_epoch = np.load(f'{save_path}val_logprob_epoch.npy')

log_prob_mean = np.array(logprob_list)

figure = plt.figure()
plt.plot(logprob_epoch,log_prob_mean)
plt.savefig(f'{save_path}val_log_prob.png')
plt.close()

log_prob_mean_sorted = np.argsort(log_prob_mean)
logprob_epoch = np.array(logprob_epoch)
lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]

#np.save(f'{save_path}val_logprob.npy', log_prob_mean)
#np.save(f'{save_path}val_logprob_epoch.npy', logprob_epoch)


log_prob_mean_sorted = np.argsort(log_prob_mean)
logprob_epoch = np.array(logprob_epoch)
lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]

print('lowest epochs', lowest_epochs)

SR_mass = SR_data[:,0]
if args.scaled_mass:
    SR_mass = scaler.transform(SR_data[:,0].reshape(-1,1))
SR_hist = np.histogram(SR_mass, bins=60, density=True)
SR_density = rv_histogram(SR_hist)

noise = torch.randn(3*len(SR_mass), n_features).to(device).float()
mass_samples_SR = SR_density.rvs(size=len(noise))
mass_samples_SR = mass_samples_SR.reshape(-1,1)
mass_samples_SR = torch.from_numpy(mass_samples_SR).to(device).float()

print('mass_samples_SR', mass_samples_SR.shape)


if args.context_embedding:
    mass_numpy = mass_samples_SR.cpu().detach().numpy()
    digitized_mass = np.digitize(mass_numpy, context_bins)
    digitized_mass = torch.from_numpy(digitized_mass).to(device).float()

mini_batch_length = len(noise)//10

# ensembled_samples = []
# ensembled_mass = []

# start_time = time.time()
# #lowest_epochs = np.arange(0,10,1)
# for i,epoch in enumerate(lowest_epochs):
#     model.load_state_dict(torch.load(f'{save_path}model_epoch_{epoch}.pth'))
#     noise_batch = noise[i*mini_batch_length:(i+1)*mini_batch_length]
#     if args.context_embedding:
#         digitized_mass_batch = digitized_mass[i*mini_batch_length:(i+1)*mini_batch_length]
#         mass_batch = mass_samples_SR[i*mini_batch_length:(i+1)*mini_batch_length]
#         _samples = sample(model, noise_batch, digitized_mass_batch, start=0.0, end=1.0)
#     else:
#         mass_batch = mass_samples_SR[i*mini_batch_length:(i+1)*mini_batch_length] 
#         _samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
#     ensembled_samples.append(_samples)
#     ensembled_mass.append(mass_batch)

# print('deleting models')
# mass = torch.cat(ensembled_mass, dim=0)
# _samples = torch.cat(ensembled_samples, dim=0)
# _samples = torch.concat([mass, _samples], axis=1)
# samples = torch.concat([_samples, torch.ones(_samples.shape[0],1).to(device)], axis=1)
# samples_inverse = inverse_transform(samples, pre_parameters).cpu().detach().numpy()
# if args.scaled_mass:
#     samples_inverse[:,0] = scaler.inverse_transform(samples_inverse[:,0].reshape(-1,1)).reshape(-1)

# end_time = time.time()
# print('time to sample', end_time-start_time)


if args.scaled_mass:
    max_mass = scaler.transform(np.array([[args.higher_mass]]))[0][0]
    min_mass = scaler.transform(np.array([[args.lower_mass]]))[0][0]
else:
    max_mass = args.higher_mass
    min_mass = args.lower_mass

print('max_mass', max_mass)
print('min_mass', min_mass)

ensembled_samples = []
ensembled_mass = []

if not args.context_embedding:

    if args.interpolation_method == 'vector':
        interp_model = Conditional_ResNet_time_embed_vector_interpolation(frequencies=args.time_frequencies, 
                        context_features=1, 
                        input_dim=n_features, device=device,
                        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                        use_batch_norm=True, 
                        dropout_probability=0.2,
                        non_linear_context=args.non_linear_context)
    elif args.interpolation_method == 'embedding':
        interp_model = Conditional_ResNet_linear_interpolation_embedding(context_frequencies=args.context_frequencies,
                                time_frequencies=args.time_frequencies,
                                context_features=1,
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                                use_batch_norm=True,
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)
    elif args.interpolation_method == 'block':
        interp_model = Conditional_ResNet_linear_interpolation_block(context_frequencies=args.context_frequencies,
                                time_frequencies=args.time_frequencies,
                                context_features=1,
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                                use_batch_norm=True,
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)
    elif args.interpolation_method == 'low_freq':
        interp_model = Conditional_ResNet_low_freq(context_frequencies=args.context_frequencies,
                                time_frequencies=args.time_frequencies,
                                context_features=1,
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                                use_batch_norm=True,
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)
    elif args.interpolation_method == 'indiv_block':
        interp_model = Conditional_ResNet_linear_interpolation_indiv_block(context_frequencies=args.context_frequencies,
                                time_frequencies=args.time_frequencies,
                                context_features=1,
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                                use_batch_norm=True,
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context,
                                interp_block=args.interp_block)


    # if args.linear_interpolation:
    #     interp_model = Conditional_ResNet_linear_interpolation_vector(context_frequencies=args.context_frequencies,
    #                                                                   time_frequencies=args.time_frequencies, 
    #                             context_features=1, 
    #                             input_dim=n_features, device=device,
    #                             hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
    #                             use_batch_norm=True, 
    #                             dropout_probability=0.2,
    #                             non_linear_context=args.non_linear_context)
    
    else:
        interp_model = Conditional_ResNet_time_embed_linear_interpolation(frequencies=args.time_frequencies, 
                                context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                use_batch_norm=True, 
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)
else:
    interp_model = Discrete_Conditional_ResNet_linear_interpolation(frequencies=args.time_frequencies,
                                context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                use_batch_norm=True, 
                                dropout_probability=0.2,
                                context_embed=SinusoidalPosEmb(dim=args.context_frequencies,theta=100))
#scaled_mass = scaler.transform(mass.cpu().detach().numpy())

start_time = time.time()
for i,file in enumerate(lowest_epochs):
    print(file)
    interp_model.load_state_dict(torch.load(f'{save_path}/model_epoch_{file}.pth'))
    #model.load_state_dict(torch.load(f'{CR_path}model_epoch_{file}.pth'))
    noise_batch = noise[i*mini_batch_length:(i+1)*mini_batch_length]
    mass_batch = mass_samples_SR[i*mini_batch_length:(i+1)*mini_batch_length]
    # scaled_batch = scaled_mass[i*mini_batch_length:(i+1)*mini_batch_length]
    if args.context_embedding:
        _samples = sample_model_interpolation_discrete(interp_model, noise_batch, 
                                                        mass_batch, start=0.0, end=1.0, 
                                                        context_bins=context_bins, 
                                                        max_mass=max_mass, min_mass=min_mass)
    else:
        if args.interpolation_method == 'low_freq':
            _samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
        elif args.interpolation_method == 'indiv_block':
            _samples = sample_model_interpolation_block(interp_model, noise_batch, mass_batch, start=0.0, 
                                                end=1.0, max_mass=max_mass, min_mass=min_mass)
        else:
            _samples = sample_model_interpolation(interp_model, noise_batch, mass_batch, start=0.0, 
                                                end=1.0, max_mass=max_mass, min_mass=min_mass)
    #samples = sample_interpolation_(model, noise_batch, mass_batch, start=0.0, end=1.0)
    ensembled_samples.append(_samples)
    ensembled_mass.append(mass_batch)

mass = torch.cat(ensembled_mass, dim=0)
_samples = torch.cat(ensembled_samples, dim=0)
_samples = torch.concat([mass, _samples], axis=1)
samples_interpolated = torch.concat([_samples, torch.ones(_samples.shape[0],1).to(device)], axis=1)
samples_inverse_interpolated = inverse_transform(samples_interpolated, pre_parameters).cpu().detach().numpy()
if args.scaled_mass:
    samples_inverse_interpolated[:,0] = scaler.inverse_transform(samples_inverse_interpolated[:,0].reshape(-1,1)).reshape(-1)

end_time = time.time()
print('time to sample interpolated', end_time-start_time)
#    np.save(f'{save_path}samples_interpolated.npy', samples_inverse_interpolated)
#    np.save(f'{save_path}samples_interpolated_preprocessed.npy', samples_interpolated.cpu().detach().numpy())
if not args.interpolation_method == 'indiv_block':
    np.save(f'{save_path}samples_interpolated_{args.interpolation_method}.npy', samples_inverse_interpolated)
    np.save(f'{save_path}samples_interpolated_preprocessed_{args.interpolation_method}.npy', samples_interpolated.cpu().detach().numpy())
else:
    np.save(f'{save_path}samples_interpolated_{args.interpolation_method}_block_{args.interp_block}.npy', samples_inverse_interpolated)
    np.save(f'{save_path}samples_interpolated_preprocessed_{args.interpolation_method}_block_{args.interp_block}.npy', samples_interpolated.cpu().detach().numpy())



# delete_paths = [f'{save_path}model_epoch_{epoch}.pth' for epoch in logprob_epoch if epoch not in lowest_epochs]

# for path in delete_paths:
#     os.remove(path)


# np.save(f'{save_path}samples.npy', samples_inverse)
# np.save(f'{save_path}samples_preprocessed.npy', samples.cpu().detach().numpy())

###############################################################################
## %%
# for i in range(0,n_features+1,1):
#     figure = plt.figure()
#     max_val = np.max(SR_data[:,i])
#     min_val = np.min(SR_data[:,i])
#     bins = np.arange(min_val, max_val, 0.03)
    
#     plt.hist(SR_data[:,i],bins=bins,density=True, histtype='stepfilled', label='data', color='gray', alpha=0.5)
#     plt.hist(samples_inverse[:,i],bins=bins,density=True,
#             histtype='step', label='samples')
#     if args.sample_interpolated:
#         plt.hist(samples_inverse_interpolated[:,i],bins=bins,density=True,
#              histtype='step', label='samples_interpolated')
#     plt.hist(_x_test[:,i][_x_test[:,-1]==0], bins=bins, density=True, histtype='step', label='true background', color='black')
#     plt.legend()
#     plt.savefig(f'{save_path}feature_{i}.png')
#     if args.wandb:
#         #wandb.log({'sic_curve': wandb.Image(figure)})
#         wandb.log({f'feature_{i}': wandb.Image(figure)})

#     plt.close()


extrabkg = np.load(f'{args.data_dir}/extrabkg.npy')
if args.baseline:
    extrabkg = np.concatenate([extrabkg[:,:5], extrabkg[:,-1].reshape(-1,1)], axis=1)
extra_bkg = preprocess_params_transform(extrabkg, pre_parameters_cpu)[:266666]


sic_score_cathode = np.load(f'{save_path}/sic_cathode.npy')
tpr_score_cathode = np.load(f'{save_path}/tpr_cathode.npy')
fpr_score_cathode = np.load(f'{save_path}/fpr_cathode.npy')

sic_score_iad = np.load(f'{save_path}/sic_iad.npy')
tpr_score_iad = np.load(f'{save_path}/tpr_iad.npy')
fpr_score_iad = np.load(f'{save_path}/fpr_iad.npy')

sic_cathode_interpolated = np.load(f'{save_path}/sic_cathode_interpolated.npy')
tpr_cathode_interpolated = np.load(f'{save_path}/tpr_cathode_interpolated.npy')
fpr_cathode_interpolated = np.load(f'{save_path}/fpr_cathode_interpolated.npy')



from sklearn.utils.class_weight import compute_sample_weight

cathode_interpolated_data = np.vstack((x_SR[:,1:-1], samples_interpolated.detach().cpu().numpy()[:,1:-1]))
cathode_interpolated_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(samples_interpolated.shape[0])])
sample_weights_cathode_interpolated = compute_sample_weight(class_weight='balanced', y=cathode_interpolated_labels)
predict_cathode_interpolated = []

for seed in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
    clf.fit(cathode_interpolated_data, cathode_interpolated_labels.reshape(-1,1), sample_weight=sample_weights_cathode_interpolated)
    predict_cathode_interpolated.append(clf.predict_proba(x_test[:,1:-1])[:,1])

predict_cathode_interpolated = np.mean(predict_cathode_interpolated, axis=0)
sic_score_cathode_interpolated , tpr_score_cathode_interpolated , fpr_score_cathode_interpolated = SIC_cut(x_test[:,-1], predict_cathode_interpolated)

if not args.interpolation_method == 'indiv_block':
    np.save(f'{save_path}sic_cathode_{args.interpolation_method}_interpolated.npy', sic_score_cathode_interpolated)
    np.save(f'{save_path}tpr_cathode_{args.interpolation_method}_new_interpolated.npy', tpr_score_cathode_interpolated)
    np.save(f'{save_path}fpr_cathode_{args.interpolation_method}_interpolated.npy', fpr_score_cathode_interpolated)
    np.save(f'{save_path}predict_cathode_{args.interpolation_method}_interpolated.npy', predict_cathode_interpolated)
else:
    np.save(f'{save_path}sic_cathode_{args.interpolation_method}_block_{args.interp_block}_interpolated.npy', sic_score_cathode_interpolated)
    np.save(f'{save_path}tpr_cathode_{args.interpolation_method}_block_{args.interp_block}_interpolated.npy', tpr_score_cathode_interpolated)
    np.save(f'{save_path}fpr_cathode_{args.interpolation_method}_block_{args.interp_block}_interpolated.npy', fpr_score_cathode_interpolated)
    np.save(f'{save_path}predict_cathode_{args.interpolation_method}_block_{args.interp_block}_interpolated.npy', predict_cathode_interpolated)





figure = plt.figure()
plt.plot(tpr_score_iad, sic_score_iad, label='IAD')
plt.plot(tpr_score_cathode, sic_score_cathode, label=f'{args.x_train} samples')
plt.plot(tpr_score_cathode_interpolated, sic_score_cathode_interpolated, label=f'{args.x_train} new interpolated')
#plt.plot(tpr_cathode_interpolated, sic_cathode_interpolated, label=f'{args.x_train} old interpolated')
plt.legend()
if not args.interpolation_method == 'indiv_block':
    plt.savefig(f'{save_path}sic_curve_{args.interpolation_method}_interpolated.png')
else:
    plt.savefig(f'{save_path}sic_curve_{args.interpolation_method}_block_{args.interp_block}_interpolated.png')
plt.close()

figure = plt.figure()
plt.plot(1/fpr_score_iad, sic_score_iad, label='IAD')
plt.plot(1/fpr_score_cathode, sic_score_cathode, label=f'{args.x_train} samples')
plt.plot(1/fpr_score_cathode_interpolated, sic_score_cathode_interpolated, label=f'{args.x_train} samples interpolated')
#plt.plot(1/fpr_cathode_interpolated, sic_cathode_interpolated, label=f'{args.x_train} old interpolated')
plt.xscale('log')
plt.legend()
if not args.interpolation_method == 'indiv_block':
    plt.savefig(f'{save_path}sic_rejection_{args.interpolation_method}.png')
else:
    plt.savefig(f'{save_path}sic_rejection_{args.interpolation_method}_block_{args.interp_block}.png')

plt.close()
# log_prob_bkg = compute_log_prob(model, testtensor[:,1:-1], testtensor[:,0].reshape(-1,1), device=device,
#                                 batch_size=20_000, method='torchdyn')


# np.save(f'{save_path}log_prob_bkg.npy', log_prob_bkg)


# np.save(f'{save_path}sic_cathode.npy', sic_score_cathode)
# np.save(f'{save_path}tpr_cathode.npy', tpr_score_cathode)
# np.save(f'{save_path}fpr_cathode.npy', fpr_score_cathode)

# np.save(f'{save_path}sic_iad.npy', sic_score_iad)
# np.save(f'{save_path}tpr_iad.npy', tpr_score_iad)
# np.save(f'{save_path}fpr_iad.npy', fpr_score_iad)

#np.save(f'{save_path}predict_iad.npy', predict_iad)
#np.save(f'{save_path}predict_cathode.npy', predict_cathode)

# if args.wandb:
#     tprs = np.linspace(0,1,100)
#     sics_iad = np.interp(tprs, tpr_score_iad, sic_score_iad)
#     sics_cathode = np.interp(tprs, tpr_score_cathode, sic_score_cathode)

#     for i in range(100):
#         wandb.log({'SIC_iad': sics_iad[i], 'SIC_cathode': sics_cathode[i]}, step=tprs[i]) 