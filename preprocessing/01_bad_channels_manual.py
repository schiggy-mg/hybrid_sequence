#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script one of the pipeline for preprocessing:
    * here the data are loaded and the bad channels are marked manually.
    * bad channels are saved and marked as 'bads' in the info structure

"""
# %%
# import os
import mne
import os.path as op
import matplotlib as mpl
import numpy as np

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
from tools_meeg import read_MI_data_noECG, read_MI_data, save_pickle, excl_breaks
mpl.use('Qt5Agg')

# --------------------------------------
# experiment: Imagined fist clenching vs reaching task 1 and 2
# --------------------------------------
# CHANGE experiment, group HERE
# --------------------------------------
experiment = '_reaching'  # for reaching task
# experiment = ''  # for imagined fist clenching task
group = 'MI' # 'active' or 'MI' for motor imagery or 'control'

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

# --------------------------------------
# directories
# --------------------------------------
raw_dir = config.paths[f'raw{experiment}']
#save_preprocessed_dir = config.paths[f'preprocessed{experiment}']

# --------------------------------------
# general settings
# --------------------------------------
iir_params = dict(order=4, ftype='butter')  # 4 for more precise filtering compared to 2
sf = 250

# --------------------------------------
# current subject and task
# --------------------------------------
subject = subjects[17]  # change number here!!!!!
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
# current subject-dependent directories
# --------------------------------------
save_resamp_fname = op.join(config.paths[f'resamp_before_preICA{experiment}'], f'subj{subject}_before_ICA{task}.fif')
save_bad_ch_fname = op.join(config.paths[f'bad_channels{experiment}'], f'subj{subject}_bad_chan{task}')
montage_file = op.join(config.paths['chanlocs'], 'FBC64-X8.bvef')

# --------------------------------------
# Read data of the subject
# --------------------------------------
if experiment == '_reaching':
    vhdr_fname = op.join(raw_dir, f'MOVE_0{subject}/EMEEG/MOVE_0{subject}{task}.vhdr')
else:
    vhdr_fname = op.join(raw_dir, f'MOVE_0{subject}_taskM.vhdr')

# !!!! check if participant had an ECG recording or not
subj_no_ECG = ['016', '026', '021', '028', '022', '025', '027']

if subject == '031' and task == '_task1':
    vhdr_fname1 = op.join(raw_dir, f'MOVE_0{subject}/EMEEG/MOVE_0{subject}{task}_1.vhdr')
    raw_orig1 = read_MI_data_noECG(subject, raw_dir, montage_file=montage_file, vhdr_fname=vhdr_fname1)
    vhdr_fname2 = op.join(raw_dir, f'MOVE_0{subject}/EMEEG/MOVE_0{subject}{task}_2.vhdr')
    raw_orig2 = read_MI_data_noECG(subject, raw_dir, montage_file=montage_file, vhdr_fname=vhdr_fname2)

    raw_orig = mne.concatenate_raws([raw_orig1, raw_orig2])

elif subject == '059' and task == '_task2':
    vhdr_fname = op.join(raw_dir, f'MOVE_0{subject}/EMEEG/MOVE_0{subject}.vhdr')
    raw_orig = read_MI_data_noECG(subject, raw_dir, montage_file=montage_file, vhdr_fname=vhdr_fname)


elif subject in subj_no_ECG:
    raw_orig = read_MI_data_noECG(subject, raw_dir, montage_file=montage_file, vhdr_fname=vhdr_fname)
else:
    raw_orig = read_MI_data(subject, raw_dir, montage_file=montage_file, vhdr_fname=vhdr_fname)

# resample data to 250 Hz
raw_resamp = raw_orig.copy()
raw_resamp.resample(sfreq=250)

sfreq_subj = int(raw_resamp.info['sfreq'])
assert (sfreq_subj == sf)

raw1 = raw_resamp.copy()
raw1.set_eeg_reference(ref_channels='average')

events_, event_id = mne.events_from_annotations(raw1)

# --------------------------------------
# filter
# --------------------------------------
raw1.filter(l_freq=0.5, h_freq=45, method='iir', iir_params=iir_params)
raw1.notch_filter(freqs=np.arange(50, 125, 50), n_jobs=config.n_jobs)

# --------------------------------------
# Cropping
# --------------------------------------
# triggers: reaching: trigger 4: start of MI
# Mi task: trigger 4 - red cross - MI for 3 s; trigger 8 - white cross - relax for 2 s
# get timing of events
stim_event_ids = {ev: event_id[ev] for ev in event_id if "Stimulus/S  4" in ev}
stim_events_ = [ev for ev in events_ if ev[2] in stim_event_ids.values()]

if experiment == '_reaching':
    tmin = stim_events_[0][0] / sf - 2  # 2s before the first MI start trigger
    tmax = stim_events_[-1][0] / sf + 3  # 3s after the last MI start trigger
else:
    tmin = stim_events_[0][0] / sf - 1.5
    tmax = tmin + 1.5 + 200 - 0.5  # 4.5s after the last MI start

raw1.crop(tmin=tmin, tmax=tmax)

# %%
# --------------------------------------
# discarding breaks in data collection in raching task
# --------------------------------------
if experiment == '_reaching':
    raw1 = excl_breaks(raw1, t_break=6, offset=tmin, sf=sf)

# --------------------------------------
# Bad channel rejection
# --------------------------------------
# plot PSD
raw1.compute_psd(fmin=0, fmax=45).plot()

# time series. mark bad channels here!
raw1.plot(theme='light')

# %% or give as an input here:
raw1.info['bads'] += ['T8']

# When you selected the bad channels:
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# %%
channels_dropped = list(set(raw1.info['bads']))

# save the fif file with bad channels --------------------------------------
save_pickle(save_bad_ch_fname, channels_dropped)
raw1.save(save_resamp_fname, overwrite=True)
