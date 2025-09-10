#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script three from the pipeline for preprocessing:
    * this is the script for manual rejectiong of ICA bad components
"""
# %%
import os.path as op
import matplotlib as mpl
import mne

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
from tools_meeg import get_ica_weights, load_pickle
mpl.use('Qt5Agg')  # for interactive plots
# mpl.use('TkAgg')

# --------------------------------------
# experiment: Imagined fist clenching vs reaching task 1 and 2
# --------------------------------------
# CHANGE experiment HERE
# --------------------------------------
experiment = '_reaching'  # for reaching task
# experiment = ''  # for imagined fist clenching task
group = 'MI' # active' or 'MI' for motor imagery or 'control'

# --------------------------------------
# Subjects
# --------------------------------------
if experiment == '_reaching' or experiment == '_timing_check':
    if group == 'MI':
        # these are the MI subjects excluding #3 & #9 who only had 400ms for the prior movement
        my_subjects = [5, 13, 16, 19, 23, 26, 29, 37, 38, 43, 45, 51, 53, 55, 104, 107, 136, 147]
    elif group == 'active':
        # these are the active subjects excluding #2, #11, #14 & #15 who only had 400ms for the prior movement
        my_subjects = [6, 10, 20, 21, 24, 28, 31, 33, 35, 41, 44, 46, 50, 52, 59, 60]
    else:
        # excluding #1 and #8 who only had 400ms for the prior movement (waited here)
        my_subjects = [12, 17, 18, 22, 25, 27, 30, 32, 34, 40, 42, 48, 49, 54, 56, 57, 58, 139]
else:
    # these are the MI subjects who performed the MI task
    my_subjects = [5, 13, 16, 19, 26, 29, 37, 38, 43, 45, 51, 53, 104, 107, 136, 147]
    # subject 23 contracted, , #55 did not do MI task

def subj_id(subj_num):
    if subj_num < 10:
        return f'00{subj_num}'
    elif subj_num < 100:
        return f'0{subj_num}'
    else:
        return str(subj_num)


# enter your subject numbers in the range:
subjects = [subj_id(s) for s in my_subjects]
n_subj = len(subjects)

# --------------------------------------
# general settings
# --------------------------------------
iir_params = dict(order=4, ftype='butter')
sf = 250
n_jobs = config.n_jobs

# --------------------------------------
# current subject and subject-dependent directories
# --------------------------------------
subject = subjects[17]
subject_id = '0' + subject
print(f'Working on data of subject {subject_id}.')

if experiment == '_reaching':
    tasks = ['_task1', '_task2', '_task3']
    task = tasks[2]  # change number here!!!!
    print(f'Working on data of {task}.')
else:
    task = ''

# %%
# --------------------------------------
# directories
# --------------------------------------
save_preprocessed_dir = config.paths[f'preprocessed{experiment}']
save_resamp_fname = op.join(config.paths[f'resamp_before_preICA{experiment}'], f'subj{subject}_before_ICA{task}.fif')
save_resamp_ica_fname = op.join(config.paths[f'raw_ica{experiment}'], f'subj{subject}_ica{task}.fif')
save_postICA_fname = op.join(config.paths[f'raw_ica{experiment}'], f'subj{subject}_postICA{task}.fif')
eogannot_fname = op.join(config.paths[f'eogannot{experiment}'], f'{subject_id}-eog-annot{task}')

# --------------------------------------
# Read data and bad channels
# --------------------------------------
raw_ica = mne.io.read_raw_fif(save_resamp_ica_fname)
raw = mne.io.read_raw_fif(save_resamp_fname)

# --------------------------------------
# read ICA
# --------------------------------------

# get ICA results
fit_params = dict(ortho=False, extended=True)
picks = mne.pick_types(raw_ica.info, eeg=True, eog=True, meg=False, exclude='bads')
n_components = len(picks) - 1
ica = get_ica_weights(subject_id,
                      raw_ica,
                      picks=picks,
                      reject=None,
                      method='picard',
                      n_components=n_components,
                      fit_params=fit_params,
                      random_state=42,  # I guess we want to keep this fixed
                      ica_from_disc=True,
                      save_to_disc=False,
                      ica_path=config.paths[f'ica{experiment}']+f'{task}')

#ica.exclude = exclude
# %%
# --------------------------------------
# plot ICA
# --------------------------------------

ica.plot_components(range(0, 32), inst=raw_ica)
ica.plot_components(range(32, ica.n_components), inst=raw_ica)

ica.plot_sources(inst=raw_ica, theme='light')
# mne.viz.plot_sensors(raw_ica.info, show_names=True)

# EXCLUDE ICA HERE!!!!
ica.exclude = [0,1,2,3,9,10,15,16,18,20,21,23,25,26,27,28,30,31]
# Now we apply these ICA weights to the original epochs:

# check the psd how the rejection worked----------------------------
raw_ica_2 = raw_ica.copy()
raw_ica_2.load_data()
ica.apply(raw_ica_2)
raw_ica_2.compute_psd(fmin=0, fmax=45).plot()

# time series
raw_ica_2.plot(theme='light')

# %%
# after making sure that all is fine: -----------------
raw_postica = ica.apply(raw.copy().load_data())
raw_postica.save(save_postICA_fname, overwrite=True)
# raw_postica.filter(l_freq=0.5, h_freq=45, method='iir', iir_params=iir_params)
# raw_postica.plot(theme='light')