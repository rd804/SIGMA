import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')
from src.generate_data_lhc import *
from src.utils import *
from matplotlib import pyplot as plt
#import pickle

nsig = 3000

log_p = []

plt.figure()

#seed = 3
for seed in [8]:
    datadir = 'data/baseline_delta_R'
    savedir = f'results/signal_scan/nsig_{nsig}/seed_{seed}/'
    savedir_CR = f'results/signal_scan_CR/nsig_{nsig}/seed_{seed}/'

    samples = np.load(savedir + 'samples.npy')
    samples_interpolated = np.load(savedir + 'samples_interpolated.npy')
    samples_CR = np.load(savedir_CR + 'samples.npy')
    samples_interpolated_vector = np.load(savedir + 'samples_interpolated_vector.npy')

    SR_data, _ , _, _ = resample_split(datadir, n_sig = nsig, resample_seed = seed,resample = True)

    pre_parameters_cpu = preprocess_params_fit(SR_data)
    data_preprocessed = preprocess_params_transform(SR_data, pre_parameters_cpu)



    data_samples_preprocessed = preprocess_params_transform(samples, pre_parameters_cpu)
    data_samples_interpolated_preprocessed = preprocess_params_transform(samples_interpolated, pre_parameters_cpu)
    data_samples_CR_preprocessed = preprocess_params_transform(samples_CR, pre_parameters_cpu)
    data_samples_interpolated_vector_preprocessed = preprocess_params_transform(samples_interpolated_vector, pre_parameters_cpu)

    x_train = np.concatenate([data_samples_preprocessed,data_samples_interpolated_preprocessed,
                            data_samples_CR_preprocessed, 
                            data_samples_interpolated_vector_preprocessed])[:,1:-1]

    labels = np.concatenate([0*np.ones(data_samples_preprocessed.shape[0]),
                             1*np.ones(data_samples_interpolated_preprocessed.shape[0]),
                             2*np.ones(data_samples_CR_preprocessed.shape[0]),
                             3*np.ones(data_samples_interpolated_vector_preprocessed.shape[0])])

    print('labels:', labels.shape)
# import one-hot encoder
#from sklearn.preprocessing import OneHotEncoder 
# instantiate one hot encoder
#enc = OneHotEncoder()
# fit and transform labels
#labels = enc.fit_transform(labels.reshape(-1,1)).toarray()
#print('encoded labels:', labels.shape)


    predictions = []    
    for ensemble in range(10):
        #X_train, X_test, y_train, y_test = train_test_split(x_train, labels, test_size=0.2, random_state=seed)
        model = HistGradientBoostingClassifier(max_iter=1000, random_state=ensemble, validation_fraction=0.5)
        model.fit(x_train, labels)
        y_pred = model.predict_proba(data_preprocessed[:,1:-1][data_preprocessed[:,-1]==0])
        print(y_pred.shape)
        predictions.append(y_pred)

    #predictions = np.array(predictions)
    #prediction = np.mean(predictions, axis=0)
    log_posterior = np.log(np.mean(predictions, axis=1))
    #log_p.append(predictions)
    #log_posterior = np.mean(np.log(predictions), axis=0)
    log_p.append(log_posterior)


plot_points = [0,1,2,3]
plot_labels = ['data','interpolated', 'CR', 'linear int vector']

log_p = np.array(log_p)[0]
print(log_p.shape)

print(np.mean(log_p, axis=0))
print(np.std(log_p, axis=0))
# scatter plot log posterior vs plot labels
#plt.scatter(plot_points, log_posterior, label='log posterior')
# plot log_p with error bars, no line
plt.errorbar(plot_points, np.mean(log_p, axis=0), yerr=np.std(log_p, axis=0), label='log posterior', fmt='o')
plt.xticks(plot_points, plot_labels)
plt.xlabel('Data type')
plt.ylabel('log posterior')
#plt.axhline(0, color='black', linestyle='--')
#plt.yscale('log')
plt.legend()
plt.savefig(f'figures/log_posterior_{nsig}_seed_{seed}.png', bbox_inches='tight')
plt.close()

