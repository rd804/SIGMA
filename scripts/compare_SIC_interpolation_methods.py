import torch
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import interp1d
import numpy.ma as ma
import argparse

args = argparse.ArgumentParser()
args.add_argument('--nsig', type=int, default=1000)
args.add_argument('--interpolation_method', type=str, default='linear')

args = args.parse_args()


def roc_interp(fpr, tpr, fpr_interp):

    tpr_interp = interp1d(fpr, tpr,fill_value=0.0,bounds_error=False)(fpr_interp)
    
    return tpr_interp, fpr_interp
 


def ensembled_SIC_at_fpr(tpr_list, fpr_list):

    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)

 #  print(fpr_list.shape, tpr_list.shape)


    tpr_median = np.nanmedian(tpr_list, axis=0)
    fpr_median = np.nanmedian(fpr_list, axis=0)
    
    tpr_max = np.nanpercentile(tpr_list, 84, axis=0)
    tpr_min = np.nanpercentile(tpr_list, 16, axis=0)

   # print(sum(tpr_max-tpr_min))


    sic_median = np.nan_to_num(tpr_median/np.sqrt(fpr_median + 1e-16), posinf=0, neginf=0, nan=0)
    sic_max = np.nan_to_num(tpr_max/np.sqrt(fpr_median+ 1e-16), posinf=0, neginf=0, nan=0)
    sic_min = np.nan_to_num(tpr_min/np.sqrt(fpr_median+ 1e-16), posinf=0, neginf=0, nan=0)

    return sic_median, sic_max, sic_min, fpr_median


savdir = 'results/signal_scan/nsig'

savedir_CR = 'results/signal_scan_CR/nsig'

nsig_list = [1000]
threshold=0.001
#nsig_list = [1000]

sic_at_fpr_iad = []
sic_at_fpr_cathode = []
sic_at_fpr_cathode_interpolated = []
sic_at_fpr_cathode_interpolated_new = []

sic_at_fpr_iad_max = []
sic_at_fpr_cathode_max = []
sic_at_fpr_cathode_interpolated_max = []
sic_at_fpr_cathode_interpolated_new_max = []

sic_at_fpr_iad_min = []
sic_at_fpr_cathode_min = []
sic_at_fpr_cathode_interpolated_min = []
sic_at_fpr_cathode_interpolated_new_min = []

sic_at_fpr_cathode_CR = []
sic_at_fpr_cathode_CR_max = []
sic_at_fpr_cathode_CR_min = []

fpr_interp = np.linspace(0.0001, 1, 10000)


for nsigs in nsig_list:

   print(f'nsig={nsigs}')
   tpr_iad_list = []
   sic_iad_list = []
   fpr_iad_list = []
   tpr_cathode_list = []
   sic_cathode_list = []
   fpr_cathode_list = []
   tpr_cathode_interpolated_list = []
   sic_cathode_interpolated_list = []
   fpr_cathode_interpolated_list = []

   tpr_cathode_new_interpolated_list = []
   sic_cathode_new_interpolated_list = []
   fpr_cathode_new_interpolated_list = []

   tpr_cathode_CR_list = []
   sic_cathode_CR_list = []
   fpr_cathode_CR_list = []

   for seed in range(10):
     # seed = 2
      directory = f'{savdir}_{nsigs}/seed_{seed}'
      print(directory)
      if not os.path.exists(f'{directory}/tpr_iad.npy'):
         continue
      tpr_iad = np.load(f'{directory}/tpr_iad.npy')
      sic_iad = np.load(f'{directory}/sic_iad.npy')
      fpr_iad = np.load(f'{directory}/fpr_iad.npy')

  #    print(np.min(tpr_iad))

      #print(tpr_iad.shape, fpr_iad.shape)

   #   print(len(tpr_iad), len(fpr_iad))

      tpr_iad, fpr_iad = roc_interp(fpr_iad.flatten(), tpr_iad.flatten(), fpr_interp)

      tpr_cathode = np.load(f'{directory}/tpr_cathode.npy')
      sic_cathode = np.load(f'{directory}/sic_cathode.npy')
      fpr_cathode = np.load(f'{directory}/fpr_cathode.npy')

      tpr_cathode, fpr_cathode = roc_interp(fpr_cathode.flatten(), tpr_cathode.flatten(), fpr_interp)

      tpr_cathode_interpolated = np.load(f'{directory}/tpr_cathode_interpolated.npy')
      sic_cathode_interpolated = np.load(f'{directory}/sic_cathode_interpolated.npy')
      fpr_cathode_interpolated = np.load(f'{directory}/fpr_cathode_interpolated.npy')

      tpr_cathode_interpolated, fpr_cathode_interpolated = roc_interp(fpr_cathode_interpolated.flatten(), tpr_cathode_interpolated.flatten(), fpr_interp)

      tpr_cathode_new_interpolated = np.load(f'{directory}/tpr_cathode_{args.interpolation_method}_new_interpolated.npy')
      sic_cathode_new_interpolated = np.load(f'{directory}/sic_cathode_{args.interpolation_method}_interpolated.npy')
      fpr_cathode_new_interpolated = np.load(f'{directory}/fpr_cathode_{args.interpolation_method}_interpolated.npy')

      tpr_cathode_new_interpolated, fpr_cathode_new_interpolated = roc_interp(fpr_cathode_new_interpolated.flatten(), tpr_cathode_new_interpolated.flatten(), fpr_interp)
      

      directory_CR = f'{savedir_CR}_{nsigs}/seed_{seed}'
      print(directory_CR)
      if not os.path.exists(f'{directory_CR}/tpr_iad.npy'):
         continue

      CR_tpr_cathode = np.load(f'{directory_CR}/tpr_cathode.npy')
      CR_sic_cathode = np.load(f'{directory_CR}/sic_cathode.npy')
      CR_fpr_cathode = np.load(f'{directory_CR}/fpr_cathode.npy')

      CR_tpr_cathode, CR_fpr_cathode = roc_interp(CR_fpr_cathode.flatten(), CR_tpr_cathode.flatten(), fpr_interp)

      tpr_cathode_CR_list.append(CR_tpr_cathode)
      sic_cathode_CR_list.append(CR_sic_cathode)
      fpr_cathode_CR_list.append(CR_fpr_cathode)

      tpr_iad_list.append(tpr_iad)
   #   sic_iad_list.append(sic_iad)
      fpr_iad_list.append(fpr_iad)
      tpr_cathode_list.append(tpr_cathode)
    #  sic_cathode_list.append(sic_cathode)
      fpr_cathode_list.append(fpr_cathode)
      tpr_cathode_interpolated_list.append(tpr_cathode_interpolated)
     # sic_cathode_interpolated_list.append(sic_cathode_interpolated)
      fpr_cathode_interpolated_list.append(fpr_cathode_interpolated)

      tpr_cathode_new_interpolated_list.append(tpr_cathode_new_interpolated)
      sic_cathode_new_interpolated_list.append(sic_cathode_new_interpolated)
      fpr_cathode_new_interpolated_list.append(fpr_cathode_new_interpolated)




   tpr_iad_list = np.array(tpr_iad_list)
   fpr_iad_list = np.array(fpr_iad_list)

   
   # for i in range(len(tpr_iad_list)):
   #    plt.plot(1/fpr_cathode_interpolated_list[i], tpr_cathode_interpolated_list[i]/np.sqrt(fpr_cathode_interpolated_list[i]))
   # plt.xscale('log')
   # plt.show()

   sic_iad, sic_iad_max, sic_iad_min, fpr_iad = ensembled_SIC_at_fpr(tpr_iad_list, fpr_iad_list)
   sic_cathode, sic_cathode_max, sic_cathode_min, fpr_cathode = ensembled_SIC_at_fpr(tpr_cathode_list, fpr_cathode_list)
   sic_cathode_interpolated, sic_cathode_interpolated_max, sic_cathode_interpolated_min, fpr_cathode_interpolated = ensembled_SIC_at_fpr(tpr_cathode_interpolated_list, fpr_cathode_interpolated_list)
   sic_cathode_CR, sic_cathode_CR_max, sic_cathode_CR_min, fpr_cathode_CR = ensembled_SIC_at_fpr(tpr_cathode_CR_list, fpr_cathode_CR_list)
   sic_cathode_new_interpolated, sic_cathode_new_interpolated_max, sic_cathode_new_interpolated_min, fpr_cathode_new_interpolated = ensembled_SIC_at_fpr(tpr_cathode_new_interpolated_list, fpr_cathode_new_interpolated_list)


   sic_at_fpr_iad.append(sic_iad[np.argmin(np.abs(fpr_iad-threshold))])
   sic_at_fpr_iad_max.append(sic_iad_max[np.argmin(np.abs(fpr_iad-threshold))])
   sic_at_fpr_iad_min.append(sic_iad_min[np.argmin(np.abs(fpr_iad-threshold))])

   sic_at_fpr_cathode.append(sic_cathode[np.argmin(np.abs(fpr_cathode-threshold))])
   sic_at_fpr_cathode_max.append(sic_cathode_max[np.argmin(np.abs(fpr_cathode-threshold))])
   sic_at_fpr_cathode_min.append(sic_cathode_min[np.argmin(np.abs(fpr_cathode-threshold))])

   sic_at_fpr_cathode_interpolated.append(sic_cathode_interpolated[np.argmin(np.abs(fpr_cathode_interpolated-threshold))])
   sic_at_fpr_cathode_interpolated_max.append(sic_cathode_interpolated_max[np.argmin(np.abs(fpr_cathode_interpolated-threshold))])
   sic_at_fpr_cathode_interpolated_min.append(sic_cathode_interpolated_min[np.argmin(np.abs(fpr_cathode_interpolated-threshold))])


   sic_at_fpr_cathode_interpolated_new.append(sic_cathode_new_interpolated[np.argmin(np.abs(fpr_cathode_new_interpolated-threshold))])
   sic_at_fpr_cathode_interpolated_new_max.append(sic_cathode_new_interpolated_max[np.argmin(np.abs(fpr_cathode_new_interpolated-threshold))])
   sic_at_fpr_cathode_interpolated_new_min.append(sic_cathode_new_interpolated_min[np.argmin(np.abs(fpr_cathode_new_interpolated-threshold))])

   sic_at_fpr_cathode_CR.append(sic_cathode_CR[np.argmin(np.abs(fpr_cathode_CR-threshold))])
   sic_at_fpr_cathode_CR_max.append(sic_cathode_CR_max[np.argmin(np.abs(fpr_cathode_CR-threshold))])
   sic_at_fpr_cathode_CR_min.append(sic_cathode_CR_min[np.argmin(np.abs(fpr_cathode_CR-threshold))])


   plt.plot(1/fpr_iad, sic_iad, label='IAD')
   plt.fill_between(1/fpr_iad, sic_iad_min, sic_iad_max, alpha=0.5)
   plt.plot(1/fpr_cathode, sic_cathode, label='Data, no interpolation')
   plt.fill_between(1/fpr_cathode, sic_cathode_min, sic_cathode_max, alpha=0.5)
   plt.plot(1/fpr_cathode_interpolated, sic_cathode_interpolated, label='Data, with interpolation')
   plt.fill_between(1/fpr_cathode_interpolated, sic_cathode_interpolated_min, sic_cathode_interpolated_max, alpha=0.5)
   plt.plot(1/fpr_cathode_CR, sic_cathode_CR, label='Interpolation from CR')
   plt.fill_between(1/fpr_cathode_CR, sic_cathode_CR_min, sic_cathode_CR_max, alpha=0.5)
   plt.plot(1/fpr_cathode_new_interpolated, sic_cathode_new_interpolated, label='New interpolation')
   plt.fill_between(1/fpr_cathode_new_interpolated, sic_cathode_new_interpolated_min, sic_cathode_new_interpolated_max, alpha=0.5)
   plt.xscale('log')
   #plt.yscale('log')
   plt.legend(frameon=False, loc='upper left')
   plt.xlabel(r'1/$\epsilon_B$', fontsize=14)
   plt.ylabel('SIC', fontsize=14)
   plt.title(rf'Baseline + $\Delta R,$ $N_{{sig}}={nsigs}$')
   plt.savefig(f'figures/sic_dR_{nsigs}_{args.interpolation_method}.pdf', bbox_inches='tight')
   plt.show()

