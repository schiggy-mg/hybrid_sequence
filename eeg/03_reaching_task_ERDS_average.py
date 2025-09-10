#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
creation of plots averaged over participants

this script creates Supplementary plots 3A and B.
'''
# %%
from matplotlib import pyplot as plt
import numpy as np
import mne
import pandas as pd
import seaborn as sns
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

from tools_meeg import *
import config

# --------------------------------------
# directories
# --------------------------------------
save_preprocessed_dir = config.paths['preprocessed_reaching']

group = 'M'
#group = 'A'
#group = 'C'

if group == 'M':
    # these are the MI subjects excluding #3 & #9 who only had 400ms for the prior movement
    my_subjects = [5, 13, 16, 19, 23, 26, 29, 37, 38, 43, 45, 51, 53, 55, 104, 107, 136, 147]
    # subj 23 contracted in MI task
elif group == 'A':
    # these are the active subjects excluding #2, #11, #14 & #15 who only had 400ms for the prior movement
    my_subjects = [6, 10, 20, 21, 24, 28, 31, 33, 35, 41, 44, 46, 50, 52, 59, 60]
else:  # control Group
    # excluding #1 and #8 who only had 400ms for the prior movement (waited here)
    my_subjects = [12, 17, 18, 22, 25, 27, 30, 32, 34, 40, 42, 48, 49, 54, 56, 57, 58, 139]

def subj_id(subj_num):
    if subj_num < 10:
        return f'00{subj_num}'
    elif subj_num < 100:
        return f'0{subj_num}'
    else:
        return str(subj_num)


subjects = [subj_id(s) for s in my_subjects]

# fixed variables
tmin = -1  # -1 for epoching to cue color change, -2 for epoching to movement onset
tmax = 2.5 # 2.5 for epoching to cue color change, 1.5 for epoching to movement onset
freqs = np.arange(5., 35.5, 0.5) # frequencies
# freqs = np.arange(13., 26)
n_cycles = freqs / 2.
n_jobs = config.n_jobs

tfr_csd_list = [None for i in range(len(subjects))]

tfr_values_min = [None for i in range(len(subjects))]
tfr_values_max = [None for i in range(len(subjects))]


for i, subject in enumerate(subjects):
    print(i, subject)

    # READ -------------------------------
    epochs1 = mne.read_epochs(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo_task1.fif')
    epochs1.interpolate_bads() # needed because otherwise channels are excluded in the grand average
    epochs1.event_id = {'Stimulus/S  4': 4}  # to make sure event ids and events are the same everywhere
    for j, event in enumerate(epochs1.events):
        epochs1.events[j][2] = 4


    epochs2 = mne.read_epochs(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo_task2.fif')
    epochs2.interpolate_bads() # needed because otherwise channels are excluded in the grand average
    epochs2.event_id = {'Stimulus/S  4': 4}
    for j, event in enumerate(epochs2.events):
        epochs2.events[j][2] = 4

    epochs = mne.concatenate_epochs([epochs1, epochs2])

    # current source density values
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    tfr_csd = mne.time_frequency.tfr_multitaper(epochs_csd, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True, decim=2, n_jobs=n_jobs)
    tfr_csd.crop(tmin, tmax)
    tfr_csd.apply_baseline(baseline=(-0.75, -0.25), mode='percent')
    # tfr_csd.apply_baseline(baseline=(tmin, tmax), mode='percent')
    tfr_csd_list[i] = tfr_csd

    # plot individual participant tfr
    # tfr_csd.plot(['C3'], vmin=-0.6, vmax=0.6)
    # tfr_csd.plot(['C4'], vmin=-0.6, vmax=0.6)

    # get the biggest ERD value (smallest value) C3, time frame 0 to 0.6s
    # alpha 8-13, beta1 = 13.5-25, beta2 = 25.5-35
    tfr_C3 = tfr_csd.copy()
    tfr_C3.pick_channels(['C3']).crop(tmin=0, tmax=0.6, fmin=8, fmax=13)
    min_ERD = tfr_C3.data.min()
    tfr_values_min[i] = min_ERD

    # get the biggest ERS value (biggest value) C3, time frame 0 to 0.6s
    tfr_C3 = tfr_csd.copy()
    tfr_C3.pick_channels(['C3']).crop(tmin=0, tmax=0.6, fmin=8, fmax=13)
    max_ERS = tfr_C3.data.max()
    tfr_values_max[i] = max_ERS

# %%
# average over participants CSD
tfr_csd = mne.grand_average(all_inst=tfr_csd_list)

# tfr topoplot
tfr_csd.plot_topo(vmin=-0.3, vmax=0.3, layout_scale=0.7, tmin=-1, tmax=4)

tfr_csd_plotting = tfr_csd.copy()
tfr_csd_plotting.pick_channels(['C3', 'C4'])

fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={"width_ratios": [10, 10, 1]})
for channel, ax in enumerate(axes[:-1]): # for each channel
    tfr_csd_plotting.plot([channel], vmin=-0.6, vmax=0.6, colorbar=False, show=False, axes=ax) # 0 - channel number
    ax.set_title(tfr_csd_plotting.ch_names[channel])
    ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
    ax.axvline(0.6, linewidth=1, color="black", linestyle=":")
    if channel != 0:
        ax.set_ylabel("")
        ax.set_yticklabels("")
fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
fig.suptitle('CSD, multitaper')
fig.savefig(op.join(config.paths['plots'], f'revision/C3_C4_CSD_average_reaching_{group}.svg'))


# topoplot MI/prior movement
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0, tmax=0.3, fmin=8, fmax=13, vmin=0, vmax=0.3,
                             unit='ERD/S [%]', title='0-0.3s, freq: 8-13')
fig.savefig(op.join(config.paths['plots'], f'revision/topoplot_{group}_prior_movement.svg'))

# topoplot beta prior movement
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0, tmax=0.6, fmin=13.5, fmax=25, vmin=-0.2, vmax=0.2,
                             unit='ERD/S [%]', title='0-0.6s, freq: 13.5-25')
fig.savefig(op.join(config.paths['plots'], f'revision/topoplot_{group}_prior_movement_beta2.svg'))


# topoplot final reaching
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0.6, tmax=1.2, fmin=8, fmax=13, vmin=-0.4, vmax=0.4,
                             unit='ERD/s [%]', title='0.6-1.2s, freq: 8-13')
fig.savefig(op.join(config.paths['plots'], f'revision/topoplot_{group}_second_movement_alpha.svg'))

fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0.6, tmax=1.2, fmin=13.5, fmax=25, vmin=-0.4, vmax=0.4,
                             unit='ERD/s [%]', title='0.6-1.2s, freq: 13.5-25')
fig.savefig(op.join(config.paths['plots'], f'revision/topoplot_{group}_second_movement_beta.svg'))


# topoplot final ERS
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=1.8, tmax=2.2, fmin=13.5, fmax=25, vmin=-0.3, vmax=0.3,
                             unit='ERD/S [%]', title='1.8-2.2s, freq: 13.5-25')

# for active:
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=1.8, tmax=2.2, fmin=13.5, fmax=25, vmin=0, vmax=0.6,
                             unit='ERD/S [%]', title='1.8-2.2s, freq: 13.5-25')
fig.savefig(op.join(config.paths['plots'], f'revision/topoplot_{group}_feedback_ERS.svg'))


# %% get behavior data (only reported this for MI group since it's correlation in the Mi time window)
df_behavior = pd.read_csv(op.join(config.paths['kinarm'], 'behavioral_measure.csv'))

# get MI or active people
df = df_behavior.loc[df_behavior.Group == group, :]

# exclude 400ms people where I do not have eeg data
if group == 'M':
    df = df.loc[(df.VP != 3) &
                (df.VP != 9) ]
elif group == 'A':
    df = df.loc[(df.VP != 2) &
                (df.VP != 11) &
                (df.VP != 14) &
                (df.VP != 15) ]

behavior = df.Behavior.values

# %% pandas
df = pd.DataFrame({'VP': my_subjects, 'ERD': tfr_values_min, 'ERS': tfr_values_max, 'behavior': behavior})
df['rebound'] = df.ERS - df.ERD

# %% correlation plot
x = df.rebound
y = df.behavior
fig, axes = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=df, x=x, y=y, color='deeppink', s=100)
plt.title(f'{np.corrcoef(x, y)[0, 1]}')
