#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source reconstruction, create forward solution, inverse modeling, TFR on source space data
"""
# %%
from __future__ import annotations
from pathlib import Path
import mne
import numpy as np
from mne.datasets import fetch_fsaverage

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
import matplotlib as mpl
mpl.use('Qt5Agg')


# %% choose frequency and iteration (2 iterations for beta because otherwise kernel crashes)
frequency = 'beta'  # 'alpha' or 'beta' or 'beta_high'
iteration = ''  # '' or '2'

save_dir = config.paths['data']

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

epochs.set_eeg_reference('average', projection=True)
epochs.apply_proj()

# Read and set the EEG electrode locations, which are already in fsaverage's
# space (MNI space) for standard_1020:
montage = mne.channels.make_standard_montage("standard_1005")
epochs.set_montage(montage)

# set up forward solution
fwd = mne.make_forward_solution(
    epochs.info, trans=trans_dir, src=src, bem=bem, eeg=True,
    mindist=5.0,
    n_jobs=None
)

n_jobs = config.n_jobs

# %% frequency
if frequency == 'alpha':
    freqs = np.arange(8, 14, 1)
elif frequency == 'beta':
    if iteration == '':
        freqs = np.arange(14, 20, 1)
    else:
        freqs = np.arange(20, 26, 1)
else:
    if iteration == '':
        freqs = np.arange(26, 31, 1)
    else:
        freqs = np.arange(31, 36, 1)

# %% start for loop over participants. Minimum Norm Estimate (MNE)

for idx, subject in enumerate(subjects):

    print("Analyzing " + subject)

    if subject != '005':
        epochs = mne.read_epochs(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo.fif')

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

    # apply tfr to the source estimate
    n_cycles = freqs / 2.
    output = 'power'

    tfr_data = mne.time_frequency.tfr_array_multitaper(stc_data, sfreq=250,
                                                       freqs=freqs, n_cycles=n_cycles,
                                                       use_fft=True, decim=2,
                                                       output=output, n_jobs=n_jobs)

    # average over epochs and remove epochs dimension: voxel x freq x time
    tfr_data = np.mean(tfr_data[:, :, :, :], axis=0, keepdims=False)

    np.save(f'{save_dir}source/{frequency}/tfr{iteration}_{subject}.npy', tfr_data)

























