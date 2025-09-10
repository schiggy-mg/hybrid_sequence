#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source reconstruction, creation of manuscript supplemantary plots

Numpy tfr data are loaded, ERD/S time windows are selected and src objects are
created. ERD/S plots and correlation plots in alpha band are created
(Supplementary Figures 2B and 4B,C,D).

"""
# %%
from __future__ import annotations
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import os.path as op
from mne.datasets import fetch_fsaverage

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
# mpl.use('Qt5Agg')

# %%
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

# fetch fsaverage files and save path
subjects_dir = fetch_fsaverage()

plots_dir = config.paths['plots']

# set root path to fsaverag files
fs_average_root_path = f'{subjects_dir}{Path("/")}bem{Path("/")}'
# os.path.join

# load bem solution, source space and transformation matrix
bem = f'{fs_average_root_path}fsaverage-5120-5120-5120-bem-sol.fif'
src_fname = f'{fs_average_root_path}fsaverage-ico-5-src.fif'
trans_dir = f'{fs_average_root_path}fsaverage-trans.fif'

# Read the source space
src = mne.read_source_spaces(src_fname)

# clean epochs path
save_preprocessed_dir = config.paths['preprocessed']

# load epochs of first participant
epochs = mne.read_epochs(save_preprocessed_dir + '/subj005/1_pre-proc_005_nointerp-epo.fif')

# set up forward solution
fwd = mne.make_forward_solution(
    epochs.info, trans=trans_dir, src=src, bem=bem, eeg=True,
    mindist=5.0,
    n_jobs=None
)

# %% create stc object to use information later

epochs.set_eeg_reference('average', projection=True)
epochs.apply_proj()

# Read and set the EEG electrode locations, which are already in fsaverage's
# space (MNI space) for standard_1020:
montage = mne.channels.make_standard_montage("standard_1005")
epochs.set_montage(montage)

# compute noise covariance matrix with random white noise
epochs_noise = epochs.copy()
epochs_noise._data = np.random.normal(loc=0.0, scale=1.0, size=epochs.get_data().shape)
noise_cov = mne.compute_covariance(epochs_noise, tmin=-1, tmax=4,  # use the baseline
                                method='empirical',
                                rank=None)

# compute inverse operator (to invert the forward solution and obtain
# source estimates from sensor data)
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd,
                                                            noise_cov)

# apply inverse operator to epochs data to get source estimates
method = "MNE"
lambda2 = 0.05
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator,
                                                lambda2, method)

# epochs are in a list -> reshape to epochs x voxel x time
stc_data_list = []
for stc in stcs:
    stc_data_list.append(stc.data)
stc_data = np.array(stc_data_list)

# %% read data
ERD_list = []
ERS_list = []

corr_1_2 = []
corr_2_3 = []
corr_3_4 = []

frq_band = 'alpha'  # or 'alpha', 'beta', 'beta_high

if frq_band == 'alpha':
    freqs = np.arange(8, 14, 1)
elif frq_band == 'beta':
    freqs = np.arange(14, 26, 1)
else:
    freqs = np.arange(26, 36, 1)

fmin = freqs.min()
fmax = freqs.max()

for idx, subject in enumerate(subjects):

    tfr_data = np.load(op.join(config.paths['data'], f'source/{frq_band}/tfr_{subject}.npy'))
    if frq_band != 'alpha':
        tfr_data2 = np.load(op.join(config.paths['data'], f'source/{frq_band}/tfr2_{subject}.npy'))
        tfr_data = np.concatenate((tfr_data, tfr_data2), axis=1)

    # baseline correction: 1. subtracting mean of baseline, 2. dividing by mean of baseline
    baseline_indices_orig = np.where((epochs.times >= -0.75) & (epochs.times <= -0.25))[0]
    baseline_indices_div = np.divide(baseline_indices_orig, 2)
    baseline_indices = np.unique(np.round(baseline_indices_div)).astype(int)

    baseline_mean = np.mean(tfr_data[:, :, baseline_indices], axis=2, keepdims=True)

    data_baseline_corrected = tfr_data - baseline_mean
    data_percent_change = data_baseline_corrected / baseline_mean * 100

    # average over band of interest (here all loaded freqs!)
    # pick freq range + remove freq dimension
    band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    data = np.mean(data_percent_change[:, band_indices, :], axis=1, keepdims=False)

    # average over time
    # ERD
    ERD_indices_orig = np.where((epochs.times >= 0.5) & (epochs.times <= 0.7))[0]
    ERD_indices_div = np.divide(ERD_indices_orig, 2)
    ERD_indices = np.unique(np.round(ERD_indices_div)).astype(int)

    ERD = np.mean(data[:, ERD_indices], axis=1, keepdims=False)

    # ERS
    ERS_indices_orig = np.where((epochs.times >= 2.5) & (epochs.times <= 3))[0]
    ERS_indices_div = np.divide(ERS_indices_orig, 2)
    ERS_indices = np.unique(np.round(ERS_indices_div)).astype(int)

    ERS = np.mean(data[:, ERS_indices], axis=1, keepdims=False)

    # correlation
    corr_indices_orig_1_2 = np.where((epochs.times >= 1) & (epochs.times <= 2))[0]
    corr_indices_div_1_2 = np.divide(corr_indices_orig_1_2, 2)
    corr_indices_1_2 = np.unique(np.round(corr_indices_div_1_2)).astype(int)
    corr_mean_1_2 = np.mean(data[:, corr_indices_1_2], axis=1, keepdims=False)

    corr_indices_orig_2_3 = np.where((epochs.times >= 2) & (epochs.times <= 3))[0]
    corr_indices_div_2_3 = np.divide(corr_indices_orig_2_3, 2)
    corr_indices_2_3 = np.unique(np.round(corr_indices_div_2_3)).astype(int)
    corr_mean_2_3 = np.mean(data[:, corr_indices_2_3], axis=1, keepdims=False)

    corr_indices_orig_3_4 = np.where((epochs.times >= 3) & (epochs.times <= 4))[0]
    corr_indices_div_3_4 = np.divide(corr_indices_orig_3_4, 2)
    corr_indices_3_4 = np.unique(np.round(corr_indices_div_3_4)).astype(int)
    corr_mean_3_4 = np.mean(data[:, corr_indices_3_4], axis=1, keepdims=False)


    # Create a new SourceEstimate object
    stc_ERD = mne.SourceEstimate(ERD, vertices=stc.vertices,
    tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)

    stc_ERS = mne.SourceEstimate(ERS, vertices=stc.vertices,
    tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)


    ERD_list.append(stc_ERD)
    ERS_list.append(stc_ERS)

    corr_1_2.append(corr_mean_1_2)
    corr_2_3.append(corr_mean_2_3)
    corr_3_4.append(corr_mean_3_4)


# convert list to numpy array
stc_ERD_array = np.array(ERD_list)
stc_ERS_array = np.array(ERS_list)

corr_1_2 = np.array(corr_1_2)
corr_2_3 = np.array(corr_2_3)
corr_3_4 = np.array(corr_3_4)

stc_all_ERD = stc_ERD_array.mean()
stc_all_ERS = stc_ERS_array.mean()

# %% ERD plot; limit values for plotting are from script 02 (alpha limits)
stc_ERD_plot = stc_all_ERD

clim_ERD = dict(kind='value',
        pos_lims=[3.8691323120016445,
                  6.975871308805065,
                  12.667357596160024])

hemi = 'lh'  # 'both', 'rh', 'lh'
if hemi == 'both':
    colorbar = True
    views = 'dorsal'
    suffix = ''
else:
    colorbar = False
    views = 'lateral'
    if hemi == 'rh':
        suffix = '_R'
    else:
        suffix = '_L'

brain_ERD = stc_ERD_plot.plot(hemi=hemi, views=views, subjects_dir=subjects_dir,
        subject='fsaverage', time_label=' ', size=(800, 800), background="white",
        colorbar=colorbar, colormap='RdBu_r', clim=clim_ERD, time_viewer=False)

brain_ERD.save_image(f'{plots_dir}supplements/src_{frq_band}_ERD_{views}{suffix}.png')


# ERS plot
stc_ERS_plot = stc_all_ERS

clim_ERS = dict(kind='value',
        pos_lims=[30, 45, 75])

clim_ERS = dict(kind='value',
        pos_lims=[24.708546917418293,
                  37.53285539964599,
                  52.256377369582644])

brain_ERS = stc_ERS_plot.plot(hemi=hemi, views=views, subjects_dir=subjects_dir,
        subject='fsaverage', time_label=' ', size=(800, 800), background="white",
        colorbar=colorbar, colormap='RdBu_r', clim=clim_ERS, time_viewer=False)

brain_ERS.save_image(f'{plots_dir}supplements/src_{frq_band}_ERS_{views}{suffix}.png')

# %% load behavioral data
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

# %% correlation
n_voxels = 20484

window = '34'  # '12' or '23' or '34'
if window == '12':
    corr_data = corr_1_2
elif window == '23':
    corr_data = corr_2_3
else:
    corr_data = corr_3_4

clim_corr = dict(kind='value',
        pos_lims=[0.24808286867278548, # these are the values from alpha 2.5-3 time window
                  0.41894097151135473,
                  0.6619679689343729])

# create correlation matrix and transform into stc object
corr_matrix = np.zeros([n_voxels])
for c in range(n_voxels):
    brain_value = corr_data[:, c]
    r = np.corrcoef(brain_value, behavior)[0, 1] # pearson correlation

    corr_matrix[c] = r

stc_corr = mne.SourceEstimate(corr_matrix, vertices=stc.vertices,
    tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)

hemi = 'rh'  # 'both', 'rh', 'lh'
if hemi == 'both':
    views = 'dorsal'
    suffix = ''
else:
    views = 'lateral'
    if hemi == 'rh':
        suffix = '_R'
    else:
        suffix = '_L'

brain_corr_pl = stc_corr.plot(hemi=hemi, views=views, subjects_dir=subjects_dir,
        subject='fsaverage', time_label=' ', size=(800, 800), background="white",
        clim=clim_corr, colorbar=False, colormap='PRGn', time_viewer=False)

brain_corr_pl.save_image(f'{plots_dir}supplements/src_{frq_band}_corr_{views}{suffix}_{window}.png')
