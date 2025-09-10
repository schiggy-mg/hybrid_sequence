#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script four from the pipeline for preprocessing:
    * epoching data +
    * this is the script for manual rejectiong of bad epochs
"""
# %%
import os.path as op
import matplotlib as mpl
import mne

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
mpl.use('Qt5Agg')

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

# --------------------------------------
# directories
# --------------------------------------
save_preprocessed_dir = config.paths[f'preprocessed{experiment}']

# --------------------------------------
# general settings
# --------------------------------------
iir_params = dict(order=4, ftype='butter')
fs = 250
# n_jobs_ar = config.n_jobs

# --------------------------------------
# current subject and subject-dependent directories
# --------------------------------------
subject = subjects[17]  # change number here
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
# saved directories and read the saved products
# --------------------------------------

# paths -------------------------------
save_raw_postICA_fname = op.join(config.paths[f'raw_ica{experiment}'], f'subj{subject}_postICA{task}.fif')

# READ -------------------------------
raw_postica = mne.io.read_raw_fif(save_raw_postICA_fname)  # after pipeline 3

# raw_postica.plot(theme='light')

# --------------------------------------
# epoch data
# --------------------------------------
if experiment == '_reaching':
    tmin, tmax = -1.5, 3
    # triggers: trigger 4 - start of MI
    # epoch only by events of kind "Stimulus/*"
    events_, event_id = mne.events_from_annotations(raw_postica)

    # change trigger #4 into 40 whenever an abort follows directly after start MI (4, 128)
    # triggers are changed when I cut out segments in the continous data
    MI_trigger_id = {ev: event_id[ev] for ev in event_id if "Stimulus/S  4" in ev}
    MI_trigger = int(list(MI_trigger_id.values())[0])

    abort_trigger_id = {ev: event_id[ev] for ev in event_id if "Stimulus/S128" in ev}
    abort_trigger = int(list(abort_trigger_id.values())[0])

    for i, event in enumerate(events_):
        if events_[i][2] == MI_trigger:
            #print(i)
            #print(events_[i][2])
            if events_[i+1][2] == abort_trigger:
                events_[i][2] = 40

    stim_event_ids = {ev: event_id[ev] for ev in event_id if "Stimulus/S  4" in ev}
    stim_events_ = [ev for ev in events_ if ev[2] in stim_event_ids.values()]
    epochs = mne.Epochs(raw_postica, stim_events_, tmin=tmin, tmax=tmax, baseline=None,
                        event_id=stim_event_ids, event_repeated='error')

    # Let's make sure that for each subject we get the correct epochs
    if task == '_task1':
        n_expected_epochs = 6 * 18 + 25 * 18
    elif task == '_task2':
        n_expected_epochs = 4 * 18 + 25 * 18
    elif task == '_task3':
        n_expected_epochs = 3 * 4 * 2 * 2
    #assert len(epochs.events) == n_expected_epochs, f'Invalid number of epochs for subject {subject} in {task}'
    print(f'{len(epochs.events)}/{n_expected_epochs}')

else:
    tmin, tmax = -1.5, 4.5
    # triggers: trigger 4 - red cross - MI for 3 s; trigger 8 - white cross - relax for 2 s

    # epoch only by events of kind "Stimulus/*"
    events_, event_id = mne.events_from_annotations(raw_postica)
    stim_event_ids = {ev: event_id[ev] for ev in event_id if "Stimulus/S  4" in ev}
    stim_events_ = [ev for ev in events_ if ev[2] in stim_event_ids.values()]
    epochs = mne.Epochs(raw_postica, stim_events_, tmin=tmin, tmax=tmax, baseline=None,
                        event_id=stim_event_ids, event_repeated='merge')

    # Let's make sure that for each subject we get exactly 1200 epochs:
    n_expected_epochs = 40
    assert len(epochs.events) == n_expected_epochs, f'Invalid number of epochs for subject {subject}'

# %%
# --------------------------------------
# manually reject bad epochs
# --------------------------------------
epochs_clean = epochs.copy()

picks = mne.pick_types(epochs_clean.info, meg=False, eeg=True, eog=True, misc=True, exclude=['A2', 'ECG'])
epochs_clean.plot(n_epochs=11, picks=picks, theme='light')

# %%
# picks = mne.pick_types(epochs_clean.info, meg=False, eeg=True, eog=False, exclude='bads')
# epochs_clean.plot_psd(fmin=0, fmax=45, picks=picks, n_jobs=config.n_jobs)

# save data ------------------------------
epochs_clean.save(save_preprocessed_dir + f'/subj{subject}/1_pre-proc_{subject}_nointerp-epo{task}.fif', overwrite=True)
