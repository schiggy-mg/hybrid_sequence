#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script two from the pipeline for preprocessing:
    * this is an automatic preprocessing step. The following steps are done for all subjects:
            * ICA

"""
import os.path as op
import matplotlib as mpl
import numpy as np
import mne

import sys
config_path = '/Users/mg/Desktop/move_git/move_MI'
sys.path.append(config_path)

import config
from tools_meeg import get_ica_weights, read_MI_data_noECG, read_MI_data, load_pickle
mpl.use('Qt5Agg')


# --------------------------------------
# experiment: Imagined fist clenching vs reaching task 1 and 2
# --------------------------------------
# CHANGE experiment HERE
# --------------------------------------
experiment = '_reaching'  # for reaching task
#experiment = ''  # for imagined fist clenching task
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
# directories
# --------------------------------------
raw_dir = config.paths[f'raw{experiment}']
save_preprocessed_dir = config.paths[f'preprocessed{experiment}']

# --------------------------------------
# general settings
# --------------------------------------
iir_params = dict(order=4, ftype='butter')
sf = 250
# fs = int(raw1.info['sfreq'])
n_jobs = config.n_jobs

if experiment == '_reaching':
    tasks = ['_task1', '_task2', '_task3']
    task = tasks[2]  # change number here!!!!
    print(f'Working on data of {task}.')
else:
    task = ''

for i_subj, subject in enumerate(subjects):
    # --------------------------------------
    # Read data of the subject
    # --> a for loop on the subjects can start from here
    # --------------------------------------
    subject_id = '0' + subject
    print('************************************')
    print(f'subject {subject_id}: subject {i_subj} of {n_subj - 1}')
    print('************************************')

    # subject-dependent directories --------------------------
    save_bad_ch_fname = op.join(config.paths[f'bad_channels{experiment}'], f'subj{subject}_bad_chan{task}')
    montage_file = op.join(config.paths['chanlocs'], 'FBC64-X8.bvef')

    save_ica_fname = op.join(config.paths[f'raw_ica{experiment}'], f'subj{subject}_ica{task}.fif')
    eogannot_fname = op.join(config.paths[f'eogannot{experiment}'], f'{subject_id}-eog-annot{task}')

    # ----------
    # Read data and bad channels
    # ----------
    if experiment == '_reaching':
        save_resamp_fname = op.join(config.paths['resamp_before_preICA_reaching'], f'subj{subject}_before_ICA{task}.fif')
        raw1 = mne.io.read_raw_fif(save_resamp_fname, preload=True)
    else:
        vhdr_fname = op.join(raw_dir, 'MOVE_0' + subject + '_taskM.vhdr')

        if subject == '016' or subject == '026':
            raw_orig = read_MI_data_noECG(subject, raw_dir, vhdr_fname=vhdr_fname, montage_file=montage_file)
        else:
            raw_orig = read_MI_data(subject, raw_dir, vhdr_fname=vhdr_fname, montage_file=montage_file)

        # resample data to 250 Hz
        raw1 = raw_orig.copy()
        raw1.resample(sfreq=250)

        raw1.set_eeg_reference(ref_channels='average')

        raw1.info['bads'] = load_pickle(save_bad_ch_fname)
        events_, event_id = mne.events_from_annotations(raw1)

        # crop
        stim_event_ids = {ev: event_id[ev] for ev in event_id if "Stimulus/S  4" in ev}
        stim_events_ = [ev for ev in events_ if ev[2] in stim_event_ids.values()]

        tmin = stim_events_[0][0] / sf - 1.5
        tmax = tmin + 1.5 + 200 - 0.5  # 4.5s after the last MI start

        raw1.crop(tmin=tmin, tmax=tmax)

    # --------------------------------------
    # before ICA
    # --------------------------------------
    # making a separate version of epochs for ICA (hp-filtered at 1Hz, otherwise same filter settings)
    if experiment == '_reaching':
        raw_ica = raw1.copy().filter(l_freq=1,
                                 h_freq=None,
                                 method='iir',
                                 iir_params=iir_params,
                                 n_jobs=n_jobs)
    else:
        raw_ica = raw1.copy().filter(l_freq=1,
                                    h_freq=45,
                                    method='iir',
                                    iir_params=iir_params,
                                    n_jobs=n_jobs). \
            notch_filter(freqs=np.arange(50, 125, 50), n_jobs=n_jobs)

    raw_ica.info['bads'] = raw1.info['bads']
    raw_ica.save(save_ica_fname, overwrite=True)

    # --------------------------------------
    # ICA
    # --------------------------------------
    print('***************************************')
    print('RUNNIG ICA')
    print('***************************************')
    # Now we run ICA
    fit_params = dict(ortho=False, extended=True)
    picks = mne.pick_types(raw_ica.info, eeg=True, eog=False, meg=False, exclude='bads')
    n_components = len(picks) - 1
    ica = get_ica_weights(subject_id,
                          raw_ica,
                          picks=picks,
                          reject=None,
                          method='picard',
                          n_components=n_components,
                          fit_params=fit_params,
                          random_state=42,
                          ica_from_disc=False,
                          save_to_disc=True,
                          ica_path=config.paths[f'ica{experiment}']+f'{task}')