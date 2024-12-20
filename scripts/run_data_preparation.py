import pandas as pd
import numpy as np
import os
import argparse
import vector

parser = argparse.ArgumentParser(
    description=("Prepare LHCO dataset."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outdir", type=str, default="preprocessed_data/",
                    help="output directory")
#parser.add_argument("--S_over_sqrtB", type=float, default=-1,
 #                   help="Signal over background ratio in the signal region.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for the mixing")
parser.add_argument("--add_deltaR", action="store_true", default=False,
                    help="add the deltaR variable")
parser.add_argument("--data_shift", type=float, default=0.,
                    help="shifting jet masses by this fraction of mjj")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

minmass = 3.3
maxmass = 3.7


def deltaR(features_df):
    j1_vec = vector.array({
        "px": np.array(features_df[["pxj1"]]),
        "py": np.array(features_df[["pyj1"]]),
        "pz": np.array(features_df[["pzj1"]]),
    })
    j2_vec = vector.array({
        "px": np.array(features_df[["pxj2"]]),
        "py": np.array(features_df[["pyj2"]]),
        "pz": np.array(features_df[["pzj2"]]),
    })
    return j1_vec.deltaR(j2_vec).flatten()


def separate_SB_SR(data, minmass, maxmass):
    innermask = (data[:, 0] > minmass) & (data[:, 0] < maxmass)
    outermask = ~innermask
    return data[innermask], data[outermask]


# the "data" containing too much signal
features = pd.read_hdf("data/raw_jets/events_anomalydetection_v2.features.h5")

# additionally produced bkg
features_extrabkg = pd.read_hdf("data/raw_jets/events_anomalydetection_qcd_extra_inneronly_features.h5")

features_sig = features[features['label'] == 1]
features_bg = features[features['label'] == 0]

# Read from data
mj1mj2_bg = np.array(features_bg[['mj1', 'mj2']])
tau21_bg = np.array(features_bg[['tau2j1', 'tau2j2']]) / (1e-5+np.array(features_bg[['tau1j1', 'tau1j2']]))
mj1mj2_sig = np.array(features_sig[['mj1', 'mj2']])
tau21_sig = np.array(features_sig[['tau2j1', 'tau2j2']]) / (1e-5+np.array(features_sig[['tau1j1', 'tau1j2']]))
mj1mj2_extrabkg = np.array(features_extrabkg[['mj1', 'mj2']])
tau21_extrabkg = np.array(features_extrabkg[['tau2j1', 'tau2j2']]) / (1e-5+np.array(features_extrabkg[['tau1j1', 'tau1j2']]))

# Sorting of mj1 and mj2:
# Identifies which column has the minimum of mj1 and mj2, and sorts it so the new array mjmin contains the 
# mj with the smallest energy, and mjmax is the one with the biggest.
mjmin_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)] 
mjmax_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
mjmin_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
mjmax_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
mjmin_extrabkg = mj1mj2_extrabkg[range(len(mj1mj2_extrabkg)), np.argmin(mj1mj2_extrabkg, axis=1)] 
mjmax_extrabkg = mj1mj2_extrabkg[range(len(mj1mj2_extrabkg)), np.argmax(mj1mj2_extrabkg, axis=1)]

# Then we do the same sorting for the taus
tau21min_bg = tau21_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)]
tau21max_bg = tau21_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
tau21min_sig = tau21_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
tau21max_sig = tau21_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
tau21min_extrabkg = tau21_extrabkg[range(len(mj1mj2_extrabkg)), np.argmin(mj1mj2_extrabkg, axis=1)]
tau21max_extrabkg = tau21_extrabkg[range(len(mj1mj2_extrabkg)), np.argmax(mj1mj2_extrabkg, axis=1)]

# Calculate mjj and collect the features into a dataset, plus mark signal/bg with 1/0
pjj_sig = (np.array(features_sig[['pxj1','pyj1','pzj1']])+np.array(features_sig[['pxj2','pyj2','pzj2']]))
Ejj_sig = np.sqrt(np.sum(np.array(features_sig[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_sig[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_sig = np.sqrt(Ejj_sig**2-np.sum(pjj_sig**2, axis=1))

pjj_bg = (np.array(features_bg[['pxj1','pyj1','pzj1']])+np.array(features_bg[['pxj2','pyj2','pzj2']]))
Ejj_bg = np.sqrt(np.sum(np.array(features_bg[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_bg[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_bg = np.sqrt(Ejj_bg**2-np.sum(pjj_bg**2, axis=1))

pjj_extrabkg = (np.array(features_extrabkg[['pxj1','pyj1','pzj1']])+np.array(features_extrabkg[['pxj2','pyj2','pzj2']]))
Ejj_extrabkg = np.sqrt(np.sum(np.array(features_extrabkg[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_extrabkg[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_extrabkg = np.sqrt(Ejj_extrabkg**2-np.sum(pjj_extrabkg**2, axis=1))

if args.add_deltaR:
    # compute DeltaR JJ
    deltaR_jj_bg = deltaR(features_bg)
    deltaR_jj_sig = deltaR(features_sig)
    deltaR_jj_extrabkg = deltaR(features_extrabkg)

    dataset_bg = np.dstack((mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000, tau21min_bg, tau21max_bg, deltaR_jj_bg, np.zeros(len(mjj_bg))))[0]
    dataset_sig = np.dstack((mjj_sig/1000, mjmin_sig/1000, (mjmax_sig-mjmin_sig)/1000, tau21min_sig, tau21max_sig, deltaR_jj_sig, np.ones(len(mjj_sig))))[0]
    dataset_extrabkg = np.dstack((mjj_extrabkg/1000, mjmin_extrabkg/1000, (mjmax_extrabkg-mjmin_extrabkg)/1000, tau21min_extrabkg, tau21max_extrabkg, deltaR_jj_extrabkg, np.zeros(len(mjj_extrabkg))))[0]
else:
    dataset_bg = np.dstack((mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000, tau21min_bg, tau21max_bg, np.zeros(len(mjj_bg))))[0]
    dataset_sig = np.dstack((mjj_sig/1000, mjmin_sig/1000, (mjmax_sig-mjmin_sig)/1000, tau21min_sig, tau21max_sig, np.ones(len(mjj_sig))))[0]
    dataset_extrabkg = np.dstack((mjj_extrabkg/1000, mjmin_extrabkg/1000, (mjmax_extrabkg-mjmin_extrabkg)/1000, tau21min_extrabkg, tau21max_extrabkg, np.zeros(len(mjj_extrabkg))))[0]

if args.data_shift != 0.:
    dataset_bg[:, 1] += args.data_shift*dataset_bg[:, 0]
    dataset_bg[:, 2] += args.data_shift*dataset_bg[:, 0]
    dataset_sig[:, 1] += args.data_shift*dataset_sig[:, 0]
    dataset_sig[:, 2] += args.data_shift*dataset_sig[:, 0]
    dataset_extrabkg[:, 1] += args.data_shift*dataset_extrabkg[:, 0]
    dataset_extrabkg[:, 2] += args.data_shift*dataset_extrabkg[:, 0]

 # Set the random seed so we get a deterministic result



# format of data_all: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), sigorbg label

#data_bg_train = dataset_bg[:len(dataset_bg)//2]
#data_bg_val = dataset_bg[len(dataset_bg)//2:2*len(dataset_bg)//3]
#data_bg_train_val = dataset_bg[:2*len(dataset_bg)//3]
#data_bg_test = dataset_bg[2*len(dataset_bg)//3:]

np.save(f'{args.outdir}/data_bg.npy', dataset_bg)
np.save(f'{args.outdir}/data_sig.npy', dataset_sig)

print('background: ', len(dataset_bg))
print('signal: ', len(dataset_sig))


np.random.seed(args.seed)
np.random.shuffle(dataset_extrabkg)


# Create test set:
n_sig = -30_000

#innerdata_train, outerdata_train = separate_SB_SR(data_train, minmass, maxmass)
#innerdata_val, outerdata_val = separate_SB_SR(data_val, minmass, maxmass)
#innerdata_test, outerdata_test = separate_SB_SR(data_test, minmass, maxmass)
innerdata_extrabkg, _ = separate_SB_SR(dataset_extrabkg, minmass, maxmass)
innerdata_extrasig, _ = separate_SB_SR(dataset_sig[n_sig:], minmass, maxmass)

np.save(f'{args.outdir}/extrabkg.npy', innerdata_extrabkg)
#innerdata_extrabkg_train = innerdata_extrabkg[:200000]
#innerdata_extrabkg_val = innerdata_extrabkg[200000:266666]
innerdata_extrabkg_test = innerdata_extrabkg[266666:]
innerdata_extrasig_test = innerdata_extrasig[-20_000:]

print('test background: ', len(innerdata_extrabkg_test)) 
print('test signal: ', len(innerdata_extrasig_test))

x_test = np.concatenate((innerdata_extrabkg_test, innerdata_extrasig_test))
y_test = np.concatenate((np.zeros(len(innerdata_extrabkg_test)), np.ones(len(innerdata_extrasig_test))))

assert (x_test[:,-1] == y_test).all()
print('test: ', x_test.shape )

print('label = 0: ', len(x_test[y_test==0]))
print('label = 1: ', len(x_test[y_test==1]))




np.save(f'{args.outdir}/x_test.npy', x_test)
np.save(f'{args.outdir}/y_test.npy', y_test)



print("saved in "+args.outdir)