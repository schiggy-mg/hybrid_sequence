#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Config file with all paths

'''

import os
import os.path as op
import psutil


if 'mg' in os.getcwd():  # mac
    path_data = '/Users/mg/Desktop/MOVE/MOVE_Analysis_MG/MI_task/'
    path_data_reaching = '/Users/mg/Desktop/MOVE/MOVE_Analysis_MG/reaching_task/'
    path_raw_data = '/Users/mg/Desktop/MOVE/MI_task_raw'
    path_chan_loc = '/Users/mg/Desktop/MPI_PhD/Project_MG1/EEG_electrode_position'
    path_kinarm = '/Users/mg/Desktop/MOVE/MOVE_Analysis_MG/KINARM/'
    path_raw_reaching = '/Volumes/MOVE_data/MOVE/'
    path_plots ='/Users/mg/Desktop/Owncloud/plots/MOVE_MI/python/MI_manuscript/'
else:  # CBS cluster
    path_data = '/data/pt_02349/'

n_cores = psutil.cpu_count(logical=False)
n_jobs = n_cores - 1


# Paths:
paths = dict()
paths['data'] = path_data
paths['data_reaching'] = path_data_reaching
paths['raw'] = path_raw_data
paths['raw_reaching'] = path_raw_reaching
paths['chanlocs'] = path_chan_loc
paths['ica'] = op.join(paths['data'], 'ica')
paths['ica_reaching'] = op.join(paths['data_reaching'], 'ica')
paths['eogannot'] = op.join(paths['data'], 'eogannot')
paths['eogannot_reaching'] = op.join(paths['data_reaching'], 'eogannot')
paths['preprocessed'] = op.join(paths['data'], 'pre-processed_data')
paths['preprocessed_reaching'] = op.join(paths['data_reaching'], 'pre-processed_data')
paths['preprocessed_reaching_movementOnset'] = op.join(paths['data_reaching'], 'pre-processed_data_movementOnset')
paths['bad_channels'] = op.join(paths['data'], 'bad_channels')
paths['bad_channels_reaching'] = op.join(paths['data_reaching'], 'bad_channels')
paths['bad_epochs_preica'] = op.join(paths['data'], 'bad_epochs_preica')
paths['resamp_before_preICA'] = op.join(paths['data'], 'resamp_before_preICA')
paths['resamp_before_preICA_reaching'] = op.join(paths['data_reaching'], 'resamp_before_preICA')
paths['raw_ica'] = op.join(paths['data'], 'raw_ica')
paths['raw_ica_reaching'] = op.join(paths['data_reaching'], 'raw_ica')
paths['epochs_postICA'] = op.join(paths['data'], 'epochs_postICA')
paths['kinarm'] = path_kinarm
paths['plots'] = path_plots