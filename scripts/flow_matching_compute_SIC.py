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
from sklearn.utils.class_weight import compute_sample_weight

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--frequencies', type=int, default=3)

parser.add_argument('--size', type=int, default=100_000)

parser.add_argument('--ensemble_size', type=int, default=50)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--data_dir', type=str, default='data/extended1', help='data directory')
parser.add_argument('--time_embedding', action='store_true', help='if time embedding is used')
parser.add_argument('--device', type=str, default='cuda:1', help='device')

parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_flow_matching')
parser.add_argument('--wandb_job_type', type=str, default='test_time_embedding')
parser.add_argument('--wandb_run_name', type=str, default='extended3')


args = parser.parse_args()
save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name


job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="r_anode", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)


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


pre_parameters_cpu = preprocess_params_fit(CR_data)
#x_train = preprocess_params_transform(CR_data, pre_parameters_cpu)

# save pre_parameters
#with open(save_path+'/pre_parameters.pkl', 'wb') as f:
 #   pickle.dump(pre_parameters_cpu, f)



_x_test = np.load(f'{args.data_dir}/x_test.npy')
x_test = preprocess_params_transform(_x_test, pre_parameters_cpu)
x_SR = preprocess_params_transform(SR_data, pre_parameters_cpu)





size=args.size

ensembled_actual_samples = []

print(f'loading samples of size {size}')
for i in range(5):
    samples = np.load(f'{save_path}_{i}/samples.npy')
    print('samples shape', samples.shape)
    print(f'save path {save_path}_{i}')
    random_idx = np.random.choice(samples.shape[0], size, replace=False)
    samples = samples[random_idx]
    ensembled_actual_samples.append(samples)

samples = np.concatenate(ensembled_actual_samples, axis=0)
np.save(f'figures/samples/samples_{args.wandb_run_name}_{args.size}.npy', samples)




ensembled_samples = []

print(f'loading samples of size {size}')
for i in range(5):
    samples = np.load(f'{save_path}_{i}/samples_preprocessed.npy')
    print('samples shape', samples.shape)
    print(f'save path {save_path}_{i}')
    random_idx = np.random.choice(samples.shape[0], size, replace=False)
    samples = samples[random_idx]
    ensembled_samples.append(samples)

samples = np.concatenate(ensembled_samples, axis=0)
np.save(f'figures/samples/samples_preprocessed_{args.wandb_run_name}_{args.size}.npy', samples)


print('samples shape', samples.shape)
 
#samples_inverse = np.load(f'{save_path}samples.npy')

## %%
# for i in range(1,n_features+1,1):
#     plt.hist(SR_data[:,i],bins=100,density=True,histtype='step')
#     plt.hist(samples_inverse[:,i],bins=100,density=True,
#             histtype='step')
#     plt.savefig(f'{save_path}feature_{i}_new.png')
#     plt.close()


extrabkg = np.load(f'{args.data_dir}/extrabkg.npy')
extra_bkg = preprocess_params_transform(extrabkg, pre_parameters_cpu)[:266666]

# sample
sample_data = np.vstack((samples,x_test[x_test[:,-1]==0]))
sample_weights = compute_sample_weight(class_weight='balanced', y=sample_data[:,-1])

sample_data_train, sample_data_val = train_test_split(sample_data, test_size=0.5, random_state=args.seed)
clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0)

clf.fit(sample_data[:,1:-1], sample_data[:,-1],sample_weight=sample_weights)
predict = clf.predict_proba(sample_data_val[:,1:-1])[:,1]

auc = roc_auc_score(sample_data_val[:,-1], predict)
print('AUC_sample_quality: ', auc)

if args.wandb:
    wandb.log({'AUC_sample_quality': auc})



iad_data = np.vstack((x_SR[:,1:-1], extra_bkg[:,1:-1]))
iad_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(extra_bkg.shape[0])])
cathode_data = np.vstack((x_SR[:,1:-1], samples[:,1:-1]))
cathode_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(samples.shape[0])])
sample_weights_cathode = compute_sample_weight(class_weight='balanced', y=cathode_labels)
sample_weights_iad = compute_sample_weight(class_weight='balanced', y=iad_labels)

predict_iad = []

for seed in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
    clf.fit(iad_data, iad_labels.reshape(-1,1), sample_weight=sample_weights_iad)
    predict_iad.append(clf.predict_proba(x_test[:,1:-1])[:,1])
# clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0)
# clf.fit(iad_data, iad_labels.reshape(-1,1), sample_weight=sample_weights_iad)
predict_iad = np.mean(predict_iad, axis=0)

sic_score_iad , tpr_score_iad , _ = SIC_cut(x_test[:,-1], predict_iad)

predict_cathode = []

for seed in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
    clf.fit(cathode_data, cathode_labels.reshape(-1,1), sample_weight=sample_weights_cathode)
    predict_cathode.append(clf.predict_proba(x_test[:,1:-1])[:,1])

predict_cathode = np.mean(predict_cathode, axis=0)
sic_score_cathode , tpr_score_cathode , _ = SIC_cut(x_test[:,-1], predict_cathode)



plt.plot(tpr_score_iad, sic_score_iad, label='IAD')
plt.plot(tpr_score_cathode, sic_score_cathode, label='Cathode')
plt.title(f'SIC curve for {args.wandb_run_name}')
plt.legend()
plt.savefig(f'figures/sic_curve_{args.wandb_run_name}_{args.size}.png')
plt.close()


np.save(f'figures/sic/sic_score_cathode_{args.wandb_run_name}_{args.size}.npy', sic_score_cathode)
np.save(f'figures/sic/tpr_score_cathode_{args.wandb_run_name}_{args.size}.npy', tpr_score_cathode)

np.save(f'figures/sic/sic_score_iad_{args.wandb_run_name}.npy', sic_score_iad)
np.save(f'figures/sic/tpr_score_iad_{args.wandb_run_name}.npy', tpr_score_iad)

# log_prob_bkg = compute_log_prob(model, testtensor[:,1:-1], testtensor[:,-1].reshape(-1,1), device=device,
#                                 batch_size=10_000)


# np.save(f'{save_path}log_prob_bkg.npy', log_prob_bkg)


# np.save(f'{save_path}sic_cathode.npy', sic_score_cathode)
# np.save(f'{save_path}tpr_cathode.npy', tpr_score_cathode)
# np.save(f'{save_path}sic_iad.npy', sic_score_iad)
# np.save(f'{save_path}tpr_iad.npy', tpr_score_iad)

# if args.wandb:
#     tprs = np.linspace(0,1,100)
#     sics_iad = np.interp(tprs, tpr_score_iad, sic_score_iad)
#     sics_cathode = np.interp(tprs, tpr_score_cathode, sic_score_cathode)

#     for i in range(100):
#         wandb.log({'SIC_iad': sics_iad[i], 'SIC_cathode': sics_cathode[i]}, step=tprs[i]) 