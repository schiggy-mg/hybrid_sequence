#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster based permutation test for correlation of behavior (error reduction or
questionnaire answer) and eeg data (imagined fist clenching task or reaching task)

Pre-processed epoched data for each participant is loaded and re-referenced to
CSD. Data is transformed into time/frequency domain. Behavior of adaptation
performance in reaching task is loaded (or questionnaire data) and correlated
with eeg data in a cluster based permutation test. Script reproduces figure 4A & B,
supplementary figure3A, and supplementary figure 5.

"""

# %%
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import numpy as np
import scipy.stats as stats
import mne
import pandas as pd
import seaborn as sns
import os.path as op

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
from tools_mne_cluster_permutation import _find_clusters
from tools_cluster_permutation import prepare_for_cluster_test, reformat_clusters, permutation_test

# %% functions

def subj_id(subj_num):
    """make string of subj nummer"""
    if subj_num < 10:
        return f'00{subj_num}'
    elif subj_num < 100:
        return f'0{subj_num}'
    else:
        return str(subj_num)


# %% load eeg data

# --------------------------------------
# experiment: Imagined fist clenching vs reaching task 1 and 2
# --------------------------------------
# CHANGE experiment and group HERE
# --------------------------------------
experiment = '_reaching'  # for reaching task
#experiment = ''  # for imagined fist clenching task
# group = 'M' for imagined fist clenching task, group has to be M! active people didnt do this task
group = 'M'
# --------------------------------------
# CHANGE correlation variable HERE
# --------------------------------------
corr_variable = 'learning'
#corr_variable = 'questionnaire'

# --------------------------------------
# Subjects
# --------------------------------------
if experiment == '_reaching':
    if group == 'M':
        # these are the MI subjects excluding #3 & #9 who only had 400ms for the prior movement
        my_subjects = [5, 13, 16, 19, 23, 26, 29, 37, 38, 43, 45, 51, 53, 55, 104, 107, 136, 147]
    elif group == 'A':
        # these are the active subjects excluding #2, #11, #14 & #15 who only had 400ms for the prior movement
        my_subjects = [6, 10, 20, 21, 24, 28, 31, 33, 35, 41, 44, 46, 50, 52, 59, 60]

else:
    # these are the MI subjects who performed the MI task
    my_subjects = [5, 13, 16, 19, 26, 29, 37, 38, 43, 45, 51, 53, 104, 107, 136, 147]
    # subject 23 contracted, , #55 did not do MI task

subjects = [subj_id(s) for s in my_subjects]

save_preprocessed_dir = config.paths[f'preprocessed{experiment}']
channel_dir = config.paths['preprocessed']

# fixed variables
if experiment == '_reaching':
    tmin = -1
    tmax = 2.5
else:
    tmin = -1
    tmax = 4
freqs = np.arange(5., 35.5, 0.5)
#freqs = np.arange(5., 35, 1)
n_cycles = freqs / 2.
n_jobs = config.n_jobs

tfr_csd_list = [None for i in range(len(subjects))]

for i, subject in enumerate(subjects):
    print(i, subject)

    # READ -------------------------------
    if experiment == '_reaching':
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
    else:
        epochs = mne.read_epochs(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo.fif')
        epochs.interpolate_bads()  # necessary for grant average to work

    # current source density values
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    tfr_csd = mne.time_frequency.tfr_multitaper(epochs_csd, freqs=freqs,
                                                n_cycles=n_cycles, n_jobs=n_jobs,
                                                use_fft=True, return_itc=False,
                                                average=True, decim=2)
    tfr_csd.crop(tmin, tmax)
    tfr_csd.apply_baseline(baseline=(-0.75, -0.25), mode='percent')
    tfr_csd_list[i] = tfr_csd

# %% get behavior data
data_folder = config.paths['kinarm']
file = 'behavioral_measure.csv'
df_behavior = pd.read_csv(f'{data_folder}{file}')

# load questionnaire data
file_quest = 'questionnaire.csv'
quest = pd.read_csv(f'{data_folder}{file_quest}', sep=';')
# merge with df
df = pd.merge(left=df_behavior, right=quest, how='inner', on='VP')

# load dwell time data
file_dwell = 'dwell_time.csv'
dwell = pd.read_csv(f'{data_folder}{file_dwell}', sep=',')
# merge with df
df = pd.merge(left=df, right=dwell, how='inner', on='VP')

# load fusion index data
file_FI = 'fusion_index.csv'
FI = pd.read_csv(f'{data_folder}{file_FI}', sep=',')
# merge with df
df = pd.merge(left=df, right=FI, how='inner', on='VP')

# keep MI people
df = df.loc[df.Group_x == group, :]

# only print this for reaching task and MI group
# correlation behavior questionnaire
print(f'correlation between questionnaire and behavior in reaching task in MI group: {np.corrcoef(df.Behavior, df.MI)[0, 1]}')
print(f'correlation between dwell time in timing check task and behavior in reaching task in MI group: {np.corrcoef(df.Behavior, df.Dwell_Time)[0, 1]}')

# %%

if group == 'M':
    if experiment == '_reaching':
        # exclude 2 people with different timing
        df = df.loc[(df.VP != 3) &
                    (df.VP != 9) ]
    else:
        # exclude 3 people where I do not have eeg data
        df = df.loc[(df.VP != 3) &
                    (df.VP != 9) &
                    (df.VP != 55)]

        # exclude subj 23 becasue they contracted
        df = df.loc[(df.VP != 23)]

elif group == 'A':
    df = df.loc[(df.VP != 2) &
                (df.VP != 11) &
                (df.VP != 14) &
                (df.VP != 15) ]

else:
    df = df.loc[(df.VP != 1) &
                (df.VP != 8) ]

behavior = df.Behavior.values
#behavior = df.MI.values  # for questionnaire values
#behavior = df.MPE_adaptation_change.values
#behavior = df.MPE_basewash_change.values
#behavior = df.Slope.values

# %% transfrom eeg data into 3D array
n_VP = len(my_subjects)
n_freq = freqs.size
n_time = len(tfr_csd.times)
n_channels = tfr_csd.info['nchan']
channel_names = tfr_csd.ch_names  # to keep the channel names

# %% channel adjacency
# read one random participant's epoched data to get adjacency of channels
epochs = mne.read_epochs(f'{channel_dir}/subj005/1_pre-proc_005_nointerp-epo.fif')

epochs.interpolate_bads()

ch_adjacency, ch_names = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

# plot adjacency
# mne.viz.plot_ch_adjacency(epochs.info, ch_adjacency, ch_names)

 # %% get all needed variables
t_obs, n_tests, adjacency, sample_shape, eeg_data, corr_matrix = prepare_for_cluster_test(tfr_csd_list, n_VP, n_channels, n_freq, n_time, ch_adjacency, behavior)

# %% label cluster 3D
p_threshold = 0.01  # 0.01
threshold = stats.t.ppf(1 - p_threshold / 2, n_VP - 2)
tail = 0  # two sided test
max_step = 1
include = None
partitions = None
t_power = 1

out = _find_clusters(t_obs, threshold=threshold, tail=tail, adjacency=adjacency,
                     max_step=max_step, include=include,
                     partitions=partitions, t_power=t_power,
                     show_info=False)
clusters, cluster_stats = out

# reformat
clusters, observed_cluster_T, observed_cluster_idx  = reformat_clusters(threshold, t_obs, sample_shape, clusters, cluster_stats, adjacency, n_tests)

# %% find strongest correlations in fist clenching task
if (experiment != '_reaching') & (corr_variable == 'learning'):
    # corr_matrix shape = channels x freq (61: 5-35 in 0.5 steps) x time (627: 5s)

    # find biggest cluster mass
    mask = np.transpose(clusters[observed_cluster_idx], (2, 1, 0))
    corr_matrix_thresh = corr_matrix.copy()
    corr_matrix_thresh[~mask] = 0

    # print biggest three cluster sums
    cluster_sums = []
    for ch in range(corr_matrix_thresh.shape[0]):
        cluster_sum = np.sum(corr_matrix_thresh[ch])
        cluster_sums.append(cluster_sum)
        if cluster_sum < -2000:
            print(f'Cluster sum of {channel_names[ch]} is {cluster_sum}')

    # channel 4 (C3) has the biggest cluster sum
    # channel 2 (F3) has the second biggest cluster sum
    # channel 26 (CP5) has the third biggest cluster sum

    # find biggest correlation in C3 and F3
    channel = 4  # 2 or 4
    corr_channel = corr_matrix[channel,:,:]
    corr_min = corr_channel.min()
    indices = np.where(corr_channel == corr_min)

    print(f'The biggest correlation in {channel_names[channel]} is {corr_min} at {indices[0]/2+5}Hz, {indices[1]/125-1}s')

# %% now we need to find out if the biggest cluster is significant! (if not TFCE)
n_permutations = 100
T_distribution = np.zeros(n_permutations)
for i, shuffle in enumerate(T_distribution):
    print(i)
    t_perm = permutation_test(behavior, n_channels, n_freq, n_time, n_VP, eeg_data)

    out_perm = _find_clusters(t_perm, threshold, tail, adjacency,
                            max_step=max_step, include=include,
                            partitions=partitions, t_power=t_power,
                            show_info=False)
    clusters_perm, cluster_stats_perm = out_perm

    permuted_cluster_T = abs(cluster_stats_perm[:]).max()

    T_distribution[i] = permuted_cluster_T

# %%  check for significance
T_distribution = np.sort(T_distribution)
alpha = 0.05
cutoff = len(T_distribution) - (len(T_distribution) * alpha) - 1  # -1 because index starts at 0
alpha_value = T_distribution[round(cutoff)]

if alpha_value < observed_cluster_T:
    print('yay, sig cluster found!')

mask = np.transpose(clusters[observed_cluster_idx], (2, 1, 0))

# calculate cluster p-value
count = np.count_nonzero(T_distribution > observed_cluster_T)
p_value = count / len(T_distribution)
# %%
# Save the distibution array to a file, and/ or load saved distributions
np.save(op.join(config.paths['data'], 'distibution_FFC.npy'), T_distribution)

# %% plot of distribution
fig = plt.figure()
axx = sns.histplot(T_distribution) #, bins=20)
plt.axvline([observed_cluster_T], color='red', linestyle='--')
plt.axvline([alpha_value], color='black')
plt.show()

# %% calculate p-value
count = np.sum(T_distribution >= observed_cluster_T)
p_value = count / len(T_distribution)
print(f'the p-value of the cluster is {p_value}')

# %% plot tfr plots of all channels with sig cluster outlined
if (experiment != '_reaching') & (corr_variable == 'learning'):
    plot_folder = config.paths['plots']

    non_mask = ~mask
    vmax = 0.8
    vmin = -0.8

    ch = 4  # C3
    #ch = 2  # F3

    for ch in range(corr_matrix.shape[0]):
        fig = plt.figure()
        ax = sns.heatmap(corr_matrix[ch], center=0, cmap='PRGn',
                        vmin=vmin, vmax=vmax, cbar=True)
        # Draw the cluster outline
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if mask[ch, i, j]:
                    if i > 0 and not mask[ch, i - 1, j]:
                        plt.plot([j - 0.5, j + 0.5], [i, i], color='black', linewidth=1)
                    if i < mask.shape[1] - 1 and not mask[ch, i + 1, j]:
                        plt.plot([j - 0.5, j + 0.5], [i + 1, i + 1], color='black', linewidth=1)
                    if j > 0 and not mask[ch, i, j - 1]:
                        plt.plot([j - 0.5, j - 0.5], [i, i + 1], color='black', linewidth=1)
                    if j < mask.shape[2] - 1 and not mask[ch, i, j + 1]:
                        plt.plot([j + 0.5, j + 0.5], [i, i + 1], color='black', linewidth=1)
        ax.invert_yaxis()
        ax.axvline([126], color='black', linestyle='dotted', linewidth=1)
        ax.axvline([501], color='black', linestyle='dotted', linewidth=1)
        ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)',
            title=f'correlation matrix behavior change and ERD/ERS, {channel_names[ch]}')
        plt.show()

    fig.savefig(f'{plot_folder}C3_correlation.png', dpi=400)
    fig.savefig(f'{plot_folder}revision/C3_correlation_control_reaching.png', dpi=400)


    # %% plot topoplots, color c = correlation values
    # put correlation data into averageTFR object like tfr_csd
    # this is for MI task
    if experiment != '_reaching':

        info = tfr_csd.info
        times = tfr_csd.times
        plot_data_corr = mne.time_frequency.AverageTFR(info=info, data=corr_matrix, times=times, freqs=freqs, nave=n_VP)

        tmin_times = [1, 2, 3]
        fmax_freqs = [35, 25, 13]
        fmin_freqs_mne = [25.5, 13.5, 8]

        vmax = 0.6
        vmin = -0.6

        # for FFC value
        #vmax = 0.7
        #vmin = -0.7

        mask_params = dict(markersize=4, markerfacecolor='white')

        fig, axs = plt.subplots(len(fmin_freqs_mne), len(tmin_times))
        for count_f, fmin_mne in enumerate(fmin_freqs_mne):
            fmax_mne = fmax_freqs[count_f]
            fmin = round((fmin_mne - 5) * 2)
            fmax = round((fmax_mne - 5) * 2) + 1  # +1 because slicing excludes upper boarder
            for count_t, tmin in enumerate(tmin_times):
                # print(count, tmin)
                tmax = tmin + 1
                mask_here = mask[:, fmin:fmax, (tmin + 1) * 125:(tmax + 1) * 125]
                mne.viz.plot_tfr_topomap(plot_data_corr, tmin=tmin, tmax=tmax,
                                        fmin=fmin_mne, fmax=fmax_mne, sensors=False,
                                        vlim=(vmin, vmax), cmap='PRGn', contours=0,
                                        colorbar=True, axes=axs[count_f, count_t],
                                        mask=mask_here, mask_params=mask_params, show=False)
                axs
        plt.show()

        fig.savefig(f'{plot_folder}several_topoplots_corr_colorbar.svg')
        # fig.savefig(f'{plot_folder}supplements/several_topoplots_corr_adaptationEffect.svg')
        # fig.savefig('f'{plot_folder}supplements/several_topoplots_corr_washoutEffect.svg')
        # fig.savefig('f'{plot_folder}supplements/several_topoplots_corr_FFC_colorbar.svg')

        #  thesholded topoplots: threshold and show only data of sig channels
        corr_matrix_thresh = corr_matrix.copy()
        corr_matrix_thresh[~mask] = 0

        info = tfr_csd.info
        times = tfr_csd.times
        plot_data_corr = mne.time_frequency.AverageTFR(info=info, data=corr_matrix_thresh, times=times, freqs=freqs, nave=n_VP)

        tmin_times = [1.5, 2, 2.5, 3, 3.5]
        fmax_freqs = [35, 25, 13]
        fmin_freqs_mne = [25.5, 13.5, 8]

        vmax = 0
        vmin = -0.55

        # for FFC measure
        #vmax = 0.55
        #vmin = 0

        mask_params = dict(markersize=4, markerfacecolor='white')

        cmap = clr.LinearSegmentedColormap.from_list('custom_purple', ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7'], N=256)
        # for FFC measure
        #cmap = clr.LinearSegmentedColormap.from_list('custom_green', ['#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b'], N=256)

        fig, axs = plt.subplots(len(fmin_freqs_mne), len(tmin_times))
        for count_f, fmin_mne in enumerate(fmin_freqs_mne):
            fmax_mne = fmax_freqs[count_f]
            fmin = round((fmin_mne - 5) * 2)
            fmax = round((fmax_mne - 5) * 2) + 1  # +1 because slicing excludes upper boarder
            for count_t, tmin in enumerate(tmin_times):
                tmax = tmin + 0.5
                mne.viz.plot_tfr_topomap(plot_data_corr, tmin=tmin, tmax=tmax,
                                        fmin=fmin_mne, fmax=fmax_mne, sensors=False,
                                        vlim=(vmin, vmax), cmap=cmap, contours=0,
                                        colorbar=False, axes=axs[count_f, count_t],
                                        mask_params=mask_params, show=False)
                axs
        plt.show()

        fig.savefig(f'{plot_folder}revision/adaptation_corr_thresh05.svg')

    # raching task topoplot
    else:
        info = tfr_csd.info
        times = tfr_csd.times
        plot_data_corr = mne.time_frequency.AverageTFR(info=info, data=corr_matrix, times=times, freqs=freqs, nave=n_VP)

        tmin_times = [0, 0.5, 1, 1.5, 2]
        fmax_freqs = [35, 25, 13]
        fmin_freqs_mne = [25.5, 13.5, 8]

        vmax = 0.6
        vmin = -0.6

        #vmax = 0.7
        #vmin = -0.7

        mask_params = dict(markersize=4, markerfacecolor='white')

        fig, axs = plt.subplots(len(fmin_freqs_mne), len(tmin_times))
        for count_f, fmin_mne in enumerate(fmin_freqs_mne):
            fmax_mne = fmax_freqs[count_f]
            fmin = round((fmin_mne - 5) * 2)
            fmax = round((fmax_mne - 5) * 2) + 1  # +1 because slicing excludes upper boarder
            for count_t, tmin in enumerate(tmin_times):
                # print(count, tmin)
                tmax = tmin + 0.5
                #mask_here = mask[:, fmin:fmax, (tmin + 1) * 125:(tmax + 1) * 125]
                mne.viz.plot_tfr_topomap(plot_data_corr, tmin=tmin, tmax=tmax,
                                        fmin=fmin_mne, fmax=fmax_mne, sensors=False,
                                        vlim=(vmin, vmax), cmap='PRGn', contours=0,
                                        colorbar=False, axes=axs[count_f, count_t],
                                         show=False) # mask=mask_here, mask_params=mask_params,
                axs
        plt.show()

        fig.savefig(f'{plot_folder}revision/topoplots_correlation_control_reaching.svg')
