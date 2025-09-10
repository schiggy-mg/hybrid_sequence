#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
creation of plots averaged over participants

this script creates plots 3A,B,C; 4C and Supplementary plot 2A.

'''
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import os.path as op
from matplotlib import pyplot as plt

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

from tools_meeg import *
import config

# --------------------------------------
# directories
# --------------------------------------
save_preprocessed_dir = config.paths['preprocessed']

my_subjects = [5, 13, 16, 19, 26, 29, 37, 38, 43, 45, 51, 53, 104, 107, 136, 147]
# I excluded subj 23 because they contracted!!!

def subj_id(subj_num):
    if subj_num < 10:
        return f'00{subj_num}'
    elif subj_num < 100:
        return f'0{subj_num}'
    else:
        return str(subj_num)


subjects = [subj_id(s) for s in my_subjects]

# fixed variables
tmin = -1
tmax = 4
freqs = np.arange(5., 35.5, 0.5) # frequencies
# freqs = np.arange(8., 13.5, 0.5)
n_cycles = freqs / 2.
n_jobs = config.n_jobs

tfr_csd_list = [None for i in range(len(subjects))]

tfr_values_min = [None for i in range(len(subjects))]
tfr_values_max = [None for i in range(len(subjects))]


for i, subject in enumerate(subjects):
    print(i, subject)

    # READ -------------------------------
    epochs = mne.read_epochs(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo.fif')
    epochs.interpolate_bads() # needed because otherwise channels are excluded in the grand average

    # current source density values
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    # epochs_csd = epochs
    tfr_csd = mne.time_frequency.tfr_multitaper(epochs_csd, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs,
                                                use_fft=True, return_itc=False, average=True, decim=2)
    # tfr_csd = mne.time_frequency.tfr_morlet(epochs_csd, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True, decim=2)

    tfr_csd.crop(tmin, tmax)
    tfr_csd.apply_baseline(baseline=(-0.75, -0.25), mode='percent')
    tfr_csd_list[i] = tfr_csd

    # plot individual participant tfr
    # tfr_csd.plot(['C3'], vmin=-0.6, vmax=0.6)
    # tfr_csd.plot(['C4'], vmin=-0.6, vmax=0.6)

    # get the biggest ERD value (smallest value) C3, time frame 0 to 2s
    # alpha 8-13, beta1 = 13.5-25, beta2 = 25.5-35
    tfr_C3 = tfr_csd.copy()
    tfr_C3.pick_channels(['C3']).crop(tmin=0, tmax=2, fmin=8, fmax=13)
    tfr_values_min[i] = tfr_C3.data.min()

    # get the biggest ERS value (biggest value) C3, time frame 2 to 4s
    tfr_C3 = tfr_csd.copy()
    tfr_C3.pick_channels(['C3']).crop(tmin=2, tmax=4, fmin=8, fmax=13)
    max_ERS = tfr_C3.data.max()
    tfr_values_max[i] = max_ERS


# average over participants CSD
tfr_csd = mne.grand_average(all_inst=tfr_csd_list)

# tfr topoplot
tfr_csd.plot_topo(vmin=-0.3, vmax=0.3, layout_scale=0.7, tmin=-1, tmax=4)

tfr_csd_plotting = tfr_csd.copy()
tfr_csd_plotting.pick_channels(['C3', 'C4'])

fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={"width_ratios": [10, 10, 1]})
for channel, ax in enumerate(axes[:-1]): # for each channel
    tfr_csd_plotting.plot([channel], vmin=-0.4, vmax=0.4, colorbar=False, show=False, axes=ax) # 0 - channel number
    ax.set_title(tfr_csd_plotting.ch_names[channel])
    ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
    ax.axvline(3, linewidth=1, color="black", linestyle=":")
    if channel != 0:
        ax.set_ylabel("")
        ax.set_yticklabels("")
fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
fig.suptitle('CSD, multitaper')
fig.savefig(op.join(config.paths['plots'], 'C3_C4_CSD_multitaper_average.svg'))

# topoplot ERD
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0.5, tmax=0.7, fmin=8, fmax=13, vmin=-0.2, vmax=0.2,
                             unit='ERD/S [%]', title='0.5-0.7s, freq: 8-13')
fig.savefig(op.join(config.paths['plots'], 'topoplot_ERD.svg'))

# topoplot ERS
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=2.5, tmax=3, fmin=8, fmax=13, vmin=0, vmax=1,
                             unit='ERD/s [%]', title='2.5-3s, freq: 8-13')
fig.savefig(op.join(config.paths['plots'], 'topoplot_ERS.svg'))


# %% for supplements:
# topoplot ERD beta
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0.5, tmax=0.7, fmin=13.5, fmax=25, vmin=-0.2, vmax=0.2,
                             unit='ERD/S [%]', title='0.5-0.7s, freq: 13.5-25')
fig.savefig(op.join(config.paths['plots'], 'supplements/topoplot_ERD_beta.svg'))
# topoplot ERD high beta
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=0.5, tmax=0.7, fmin=25.5, fmax=35, vmin=-0.2, vmax=0.2,
                             unit='ERD/S [%]', title='0.5-0.7s, freq: 25.5-35')
fig.savefig(op.join(config.paths['plots'], 'supplements/topoplot_ERD_beta_high.svg'))

# topoplot ERS beta
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=2.5, tmax=3, fmin=13.5, fmax=25, vmin=-0.3, vmax=0.3,
                             unit='ERD/s [%]', title='2.5-3s, freq: 13.5-25')
fig.savefig(op.join(config.paths['plots'], 'supplements/topoplot_ERS_beta.svg'))
# topoplot ERS high beta
fig = mne.viz.plot_tfr_topomap(tfr_csd, tmin=2.5, tmax=3, fmin=25.5, fmax=35, vmin=-0.3, vmax=0.3,
                             unit='ERD/s [%]', title='2.5-3s, freq: 25.5-35')
fig.savefig(op.join(config.paths['plots'], 'supplements/topoplot_ERS_beta_high.svg'))

# %% plot line plots of ERD
df = tfr_csd_plotting.to_data_frame(time_format=None, long_format=True)

# Map to frequency bands
df['band'] = 'alpha 8-12Hz'
df = df.loc[df['freq'] >= 8] #drop freq < 8
df = df.loc[df['freq'] <= 35] #drop freq > 30

mask_beta1 = df['freq'] >= 13.5
mask_beta2 = df['freq'] >= 25.5

# df.loc[mask_beta, 'band'] = 'beta 13-30Hz'
df.loc[mask_beta1, 'band'] = 'beta_13.5-25'
df.loc[mask_beta2, 'band'] = 'beta_25.5-35'

g = sns.FacetGrid(df, row='band', margin_titles=True)
g.map(sns.lineplot, 'time', 'value', 'channel', n_boot=100, estimator='mean',
      ci=95, palette=['darkorange', 'black'])  # confidence interval 0.95
axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
g.map(plt.axhline, y=0, **axline_kw)
g.map(plt.axvline, x=0, **axline_kw)
g.map(plt.axvline, x=3, **axline_kw)
g.set(ylim=(-0.3, 0.3))
g.set_axis_labels("Time (s)", "Power change (%)")
g.set_titles(row_template="{row_name}", template='test')
g.add_legend(ncol=1, loc='upper right')
g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
g.fig.suptitle('laplacian, multitaper')
g.savefig(op.join(config.paths['plots'], 'lineplot_ERS.svg'))


# %% get behavior data
df_behavior = pd.read_csv(op.join(config.paths['kinarm'], 'behavioral_measure.csv'))

# get MI people
df = df_behavior.loc[df_behavior.Group == 'M', :]

# exclude 3 people where I do not have eeg data
df = df.loc[(df.VP != 3) &
            (df.VP != 9) &
            (df.VP != 55)]

# exclude subj 23 becasue they contracted
df = df.loc[(df.VP != 23)]

behavior = df.Behavior.values
#behavior = df.MPE_adaptation_change.values
#behavior = df.MPE_basewash_change.values
#behavior = df.Slope.values

# %% pandas df
df = pd.DataFrame({'VP': my_subjects, 'ERD': tfr_values_min, 'ERS': tfr_values_max,
                   'behavior': behavior})
df['rebound'] = df.ERS - df.ERD

# %% correlation plot
x = df.ERS
y = df.behavior
fig, axes = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=df, x=x, y=y, color='deeppink', s=100)
plt.title(f'{np.corrcoef(x, y)[0, 1]}')

plt.xlim(-0.2, 1.8)
plt.ylim(-7, -1)
fig.savefig(op.join(config.paths['plots'], 'scatter_8-13Hz_ERS_2-4s.svg'))
