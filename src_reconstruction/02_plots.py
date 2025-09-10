#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source reconstruction, creation of manuscript plots

Numpy tfr data are loaded, ERD/S time windows are selected and src objects are
created. ERD/S plots and correlation plots in alpha band are created
(Figure 3D and 4D,E). In addition, regions which are activated most are checked
with the DK atlas

"""
# %%
from __future__ import annotations
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
import os.path as op
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

# set root path to fsaverag files
fs_average_root_path = f'{subjects_dir}{Path("/")}bem{Path("/")}'

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
corr_list = []

freq_interest = 'alpha'

if freq_interest == 'alpha':
    freqs = np.arange(8., 14, 1)
    fmin = freqs.min()
    fmax = freqs.max()

for idx, subject in enumerate(subjects):

    if freq_interest == 'alpha':
        tfr_data = np.load(op.join(config.paths['data'], f'source/alpha/tfr_{subject}.npy'))

      # %%
    # baseline correction: 1. subtracting mean of baseline, 2. dividing by mean of baseline
    baseline_indices_orig = np.where((epochs.times >= -0.75) & (epochs.times <= -0.25))[0]
    baseline_indices_div = np.divide(baseline_indices_orig, 2)
    baseline_indices = np.unique(np.round(baseline_indices_div)).astype(int)

    baseline_mean = np.mean(tfr_data[:, :, baseline_indices], axis=2, keepdims=True)

    data_baseline_corrected = tfr_data - baseline_mean
    data_percent_change = data_baseline_corrected / baseline_mean * 100

    # %%
    if freq_interest == 'alpha':
        # average over alpha band
        # pick freq range + remove freq dimension
        alpha_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        data_alpha = np.mean(data_percent_change[:, alpha_indices, :], axis=1, keepdims=False)

        # average over time
        # ERD
        ERD_indices_orig = np.where((epochs.times >= 0.5) & (epochs.times <= 0.7))[0]
        ERD_indices_div = np.divide(ERD_indices_orig, 2)
        ERD_indices = np.unique(np.round(ERD_indices_div)).astype(int)

        ERD = np.mean(data_alpha[:, ERD_indices], axis=1, keepdims=False)

        # ERS should be 2.5 to 3 (correlation also plotted 3 to 3.5)
        ERS_indices_orig = np.where((epochs.times >= 2.5) & (epochs.times <= 3))[0]
        ERS_indices_div = np.divide(ERS_indices_orig, 2)
        ERS_indices = np.unique(np.round(ERS_indices_div)).astype(int)

        ERS = np.mean(data_alpha[:, ERS_indices], axis=1, keepdims=False)

        # Create a new SourceEstimate object
        stc_ERD = mne.SourceEstimate(ERD, vertices=stc.vertices,
        tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)

        stc_ERS = mne.SourceEstimate(ERS, vertices=stc.vertices,
        tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)


        ERD_list.append(stc_ERD)
        ERS_list.append(stc_ERS)
        corr_list.append(ERS)

# convert list to numpy array
if freq_interest == 'alpha':
    stc_ERD_array = np.array(ERD_list)
    stc_ERS_array = np.array(ERS_list)
    brain_corr = np.array(corr_list)

    stc_all_ERD = stc_ERD_array.mean()
    stc_all_ERS = stc_ERS_array.mean()


# %% ERD plot, only for alpha
if freq_interest == 'alpha':
    stc_ERD_plot = stc_all_ERD

    clim_ERD = dict(kind='value',
            pos_lims=[np.percentile(abs(stc_ERD_plot.data), 50),
                    np.percentile(abs(stc_ERD_plot.data), 80),
                    np.percentile(abs(stc_ERD_plot.data), 99)])

    brain_ERD = stc_ERD_plot.plot(hemi='rh', views='lateral', subjects_dir=subjects_dir,
            subject='fsaverage', time_label=' ', size=(800, 800), background="white",
            colorbar=False, colormap='RdBu_r', clim=clim_ERD, time_viewer=False)
    #brain_ERD.add_annotation("aparc", borders=2)

    brain_ERD.save_image(op.join(config.paths['plots'], 'src_alpha_ERD_lateral_R.png')
)

    # hemi='rh', 'lh', 'both'
    # views= 'lateral' (colorbar=False), 'dorsal' (colorbar=True)

# %% ERS plot
    stc_ERS_plot = stc_all_ERS

    clim_ERS = dict(kind='value',
            pos_lims=[np.percentile(abs(stc_ERS_plot.data), 50),
                    np.percentile(abs(stc_ERS_plot.data), 80),
                    np.percentile(abs(stc_ERS_plot.data), 99)])

    brain_ERS = stc_ERS_plot.plot(hemi='both', views='dorsal', subjects_dir=subjects_dir,
            subject='fsaverage', time_label=' ', size=(800, 800), background="white",
            colorbar=True, colormap='RdBu_r', clim=clim_ERS, time_viewer=False)

    brain_ERS.save_image(op.join(config.paths['plots'], 'src_alpha_ERS_dorsal.png')
)

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
if freq_interest == 'alpha':
    brain_corr_data = brain_corr

n_voxels = 20484
corr_matrix = np.zeros([n_voxels])

for c in range(n_voxels):
    ERS_value = brain_corr_data[:, c]
    r = np.corrcoef(ERS_value, behavior)[0, 1] # pearson correlation

    corr_matrix[c] = r


stc_corr = mne.SourceEstimate(corr_matrix, vertices=stc.vertices,
    tmin=stc.tmin, tstep=stc.tstep*2, subject=stc.subject)

if freq_interest == 'alpha':
    clim_corr = dict(kind='value',
            pos_lims=[np.percentile(abs(stc_corr.data), 50), # write down this value for manuscript
                    np.percentile(abs(stc_corr.data), 80),
                    np.percentile(abs(stc_corr.data), 99)])
else:  # use this also for ERS time window 3-3.5
    clim_corr = dict(kind='value',
            pos_lims=[0.24808286867278548, # these are the values from alpha 2.5-3 time window
                    0.41894097151135473,
                    0.6619679689343729])

brain_corr_pl = stc_corr.plot(hemi='both', views='lateral', subjects_dir=subjects_dir,
        subject='fsaverage', time_label=' ', size=(800, 800), background="white",
        clim=clim_corr, colorbar=False, colormap='PRGn', time_viewer=False)

brain_corr_pl.save_image(op.join(config.paths['plots'], 'src_alpha25-3_corr_lateral_R.png'))

# %%
# find ROI accoring to atlas
# I check the x most extreme ERD/ERS values for the ERD/ERS time windows respectively
n_vert_lh = src[0]['vertno'].shape[0]

ERD_data = np.squeeze(stc_all_ERD.data)
sorted_indices_ERD = np.argsort(ERD_data)
smallest_indices = sorted_indices_ERD[:20]

ERS_data = np.squeeze(stc_all_ERS.data)
sorted_indices = np.argsort(ERS_data)
biggest_indices = sorted_indices[-20:] # ERS

corr_data = np.squeeze(stc_corr.data)
sorted_indices_corr = np.argsort(corr_data)
smallest_indices = sorted_indices_corr[:20]
# Desikan/Killiany aparc
# Destrieux Atlas: https://surfer.nmr.mgh.harvard.edu/fswiki/DestrieuxAtlasChanges
# aparc.a2009s

indice_of_interest = smallest_indices

anat_label = mne.read_labels_from_annot("fsaverage", parc="aparc",
                                        subjects_dir=subjects_dir, hemi='both')

for label in anat_label:
    #print(label.hemi)
    for value in indice_of_interest:
        if value in label.vertices:
            if value <= n_vert_lh:  # belongs to left side
                if label.hemi == 'lh':
                    print(label)
            elif value > n_vert_lh:
                value = value - n_vert_lh
                if label.hemi == 'rh':
                    print(label)