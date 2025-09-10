#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
EEG functions
'''

import numpy as np
import os
import os.path as op
import mne
from mne.preprocessing import create_eog_epochs


def save_pickle(file, var):
    import pickle
    with open(file, "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(file):
    import pickle
    with open(file, "rb") as input_file:
        var = pickle.load(input_file)
    return var


def chkmk_dir(path):
    if not op.exists(path):
        os.makedirs(path)
        print('creating dir: ' + path)


def read_MI_data(subject, data_dir, montage_file, vhdr_fname, on_missing='ignore'):
    # vhdr_fname = op.join(data_dir, 'MOVE_0' + subject + '_taskM.vhdr')
    raw = mne.io.read_raw_brainvision(vhdr_fname, preload=True)
    raw = mne.set_bipolar_reference(raw, 'IO', 'SO', ch_name='VEOG')
    raw = mne.set_bipolar_reference(raw, 'H1', 'H2', ch_name='HEOG')
    raw.info['bads'] = ['A2', 'Ef', 'Ee', 'Sadd', 'Sab', 'ECG']
    ch_type_dict = {'VEOG': 'eog', 'HEOG': 'eog', 'A2': 'misc', 'Ef': 'misc', 'Ee': 'misc', 'Sadd': 'misc', 'Sab': 'misc', 'ECG': 'misc'}
    raw.set_channel_types(ch_type_dict)
    montage = mne.channels.read_custom_montage(montage_file)
    raw.set_montage(montage)
    mapping = {"AF3'": 'AF3',
               "AF4'": 'AF4',
               "PO3'": 'PO3',
               "PO4'": 'PO4',
               "AFz'": 'AFz'}
    raw.rename_channels(mapping)
    return raw


def read_MI_data_noECG(subject, data_dir, montage_file, vhdr_fname, on_missing='ignore'):
    # vhdr_fname = op.join(data_dir, 'MOVE_0' + subject + '_taskM.vhdr')
    raw = mne.io.read_raw_brainvision(vhdr_fname, preload=True)
    raw = mne.set_bipolar_reference(raw, 'IO', 'SO', ch_name='VEOG')
    raw = mne.set_bipolar_reference(raw, 'H1', 'H2', ch_name='HEOG')
    raw.info['bads'] = ['A2', 'Ef', 'Ee', 'Sadd', 'Sab']
    ch_type_dict = {'VEOG': 'eog', 'HEOG': 'eog', 'A2': 'misc', 'Ef': 'misc', 'Ee': 'misc', 'Sadd': 'misc', 'Sab': 'misc'}
    raw.set_channel_types(ch_type_dict)
    montage = mne.channels.read_custom_montage(montage_file)
    raw.set_montage(montage)
    mapping = {"AF3'": 'AF3',
               "AF4'": 'AF4',
               "PO3'": 'PO3',
               "PO4'": 'PO4',
               "AFz'": 'AFz'}
    raw.rename_channels(mapping)
    return raw


def reject_bad_segs(raw, min, max):
    """ This function rejects all time spans betweem min and max"""
    raw_segs = []
    raw_segs.append(
        raw.copy().crop( # this retains raw between beginning and min
            tmin=raw.times[0],
            tmax=min,
            include_tmax=False,
        )
    )
    raw_segs.append(
        raw.copy().crop( # this retains raw between max and end
            tmin=max,
            tmax=raw.times[-1],
            include_tmax=False,
        )
    )
            #print(tmin, tmax, len(raw_segs))
    return mne.concatenate_raws(raw_segs)


def excl_breaks(raw, t_break, offset, sf):
    """This function excludes all time spans with no trigger for more than t_break
    t_break: min time in which no trigger occures (in s)
    offset: if raw does not start at index 0, this is the offset (in s)
    (this happens if the begging was cropped)
    sf: sample frequency"""
    events_, event_id = mne.events_from_annotations(raw)
    # event timing changed due to cropping
    events_[:, 0] = events_[:, 0] - (offset * sf)
    diff = np.diff(events_, axis=0)

    while len(np.where(diff[:, 0] > t_break * sf)[0]) != 0:
            last_trigger = np.where(diff[:, 0] > t_break * sf)[0][0]
            first_trigger = last_trigger + 1

            t_last_trigger = events_[last_trigger, 0] / sf
            if events_[first_trigger][2] == 99999:  # check if first trigger is "New Segment/"
                first_trigger = last_trigger + 2
            t_first_trigger = events_[first_trigger, 0] / sf

            t_cut1 = t_last_trigger + 3
            t_cut2 = t_first_trigger - 2

            raw =  reject_bad_segs(raw, t_cut1, t_cut2)

            events_, event_id = mne.events_from_annotations(raw)
            # event timing changed due to cropping
            events_[:, 0] = events_[:, 0] - (offset * sf)

            diff = np.diff(events_, axis=0)
    return raw


def get_ica_weights(subID,
                    data_=None,
                    picks=None,
                    reject=None,
                    method='picard',
                    n_components=None,
                    fit_params=None,
                    random_state=42,
                    ica_from_disc=False,
                    save_to_disc=True,
                    ica_path=None):
    ### Load ICA data
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(ica_path, subID + '-ica.fif'))
    else:
        # data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method=method,
                                    fit_params=fit_params,
                                    random_state=random_state,
                                    n_components=n_components)
        ica.fit(data_,
                picks=picks)

        if save_to_disc:
            ica.save(fname=op.join(ica_path, subID + '-ica.fif'),
                     overwrite=True)
    return ica


