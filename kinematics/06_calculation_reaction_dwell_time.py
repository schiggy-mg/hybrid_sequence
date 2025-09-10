# -*- coding: utf-8 -*-
"""
pre-processing of data. calculation of reaction time and dwell time in task 3
(moved cues task)

annoyingly I have to get the data from the mat files (could be integrated in
script 02)

Prerequisite: mat files are already created with matlab,
(the data is filtered and kinematic data is added)

- reaction time is time until participants left the cue traget
- dwell time is time from when the middle target is entered until it is left

In this script I create and save csv files for each participant:
        1.) Generalization csv
            in Generalization_Blocks/{vp}{group}{version}Generalization.csv
            (to compare reaction and dwell time between groups)


"""

from os import listdir
import scipy.io as sio
import numpy as np
import pandas as pd

# loop over all participants
data_folder = '/data/pt_02349/MOVE/'
folder = listdir(data_folder)
folder
#folder.remove('.DS_Store')


for p in range(len(folder)):
    current_folder = folder[p]
    vp = folder[p][-4:]

    # C3D Import
    # define kinarm data path with unzipped .mat files
    subject_path = data_folder + current_folder + '/KINARM/'
    files = listdir(subject_path)
    files

    vpk = vp + '*' + 'MPE'

    print('todo ' + vpk)

    if vp == '0136':  # VP 136 has 2 mat files
        nr_file = files.index('data_kine2.mat')

        # import data file via scipy
        c3d = sio.loadmat(subject_path + files[nr_file])

        # extracting the kinarm dataframe; look at key in c3d
        mdata = c3d['data_kine2']
    else:
        # make sure to select matlab file
        nr_file = files.index('data_kine.mat')

        # import data file via scipy
        c3d = sio.loadmat(subject_path + files[nr_file])

        # extracting the kinarm dataframe; look at key in c3d
        mdata = c3d['data_kine']

    datensatz, nTotalTrials = mdata.shape

    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"

    # which condition is done (group and  version)
    cond = mdata['EXPERIMENT'][0][0][0][0]['TASK_PROTOCOL'][0]
    # c = control, M = motor imagery, a = active
    group = cond[0]
    version = cond[-7]

    # to later merge several trials in one dataframe
    appended_df = []

    # find out x and y offset (to have the middle at 0/0 in coordinate system)
    x_offset = (mdata['TARGET_TABLE'][0][0][0][0]['X_GLOBAL'][1][0]) / 100
    y_offset = (mdata['TARGET_TABLE'][0][0][0][0]['Y_GLOBAL'][1][0]) / 100

    if vp == '0136':
        x_offset2 = x_offset
        y_offset2 = y_offset
    else:
        # find offset from task 2 (in case re-calibration happend at some point)
        # I am looking at the last trial in task 2 for this (if participants had no error trial, so not really last trial)
        x_offset2 = (mdata['TARGET_TABLE'][0][1079][0][0]['X_GLOBAL'][1][0]) / 100
        y_offset2 = (mdata['TARGET_TABLE'][0][1079][0][0]['Y_GLOBAL'][1][0]) / 100

    # extracting the trials of the structure
    for trial in np.arange(nTotalTrials):
        # scipy reads in structures as structured NumPy arrays of dtype object
        # the size of the array is the size of the structure array, not the number
        # elements in any particular field. The shape defaults to 2-dimensional.
        # for convenience make a dictionary of the data using the names from dtypes
        # since the structure has only one element, but is 2-D, index it at [0, 0]
        ndata = {n: mdata[n][0, trial] for n in mdtype.names}

        # intervals are the sampling points of the measurement
        intervals = len(ndata['Right_HandX'])
        ms = np.arange(0, intervals)

        # reconstruct the columns of the data table from just the time series
        # use the number of intervals to test if a field is a column or metadata
        columns = [n for n, v in ndata.items() if v.size == intervals]

        # now make a data frame, setting the time stamps as the index
        df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)

        # adding columns
        df['Milliseconds'] = ms
        df['Trial'] = trial
        df['Events'] = 0
        df['Block_Run_Count'] = 0
        df['Cue_Target'] = 0

        # add events
        eventmetalist = ndata['EVENTS']
        eventtype = eventmetalist.dtype

        eventlist = {n: eventmetalist[n] for n in eventtype.names}
        events = eventlist['TIMES']

        events = events[0, 0][0, :]
        # change scale to ms
        events = events * 1000

        # for each event change events column to 1, 2, 3...
        eventcount = 0
        for i in events:  # add number of position in list to events column
            df.loc[df['Milliseconds'] >= round(i), 'Events'] = eventcount
            eventcount = eventcount + 1

        # save label of event
        # get list of events for each trial
        eventlabel = eventlist['LABELS']
        eventlabellist = []
        for i in range(len(eventlabel[0, 0][0])):
            eventlabellist.append(eventlabel[0, 0][0][i][0])

        eventlabels = []
        for i in df.Events:
            eventlabels.append(eventlabellist[i])

        df['Event_Label'] = eventlabels

        # add experiment information
        experimentmetalist = ndata['EXPERIMENT']
        experimenttype = experimentmetalist.dtype
        experimentlist = {n: experimentmetalist[n] for n in experimenttype.names}

        # protocol
        taskprotocol = experimentlist['TASK_PROTOCOL']
        taskprotocol = taskprotocol[0, 0][0]
        df['Task_Protocol'] = taskprotocol
        part = taskprotocol[-5:]

        # add trial information: 1.) if trial was aborted, 2.) info from TP (which cue target combi)
        trialmetalist = ndata['TRIAL']
        trialtype = trialmetalist.dtype

        triallist = {n: trialmetalist[n] for n in trialtype.names}

        # for TP
        tp = triallist['TP']
        tp = tp[0, 0][0, 0]

        df['TP'] = tp

        # info if trial was aborted
        is_error = triallist['IS_ERROR']
        is_error = is_error[0, 0][0, 0]
        df['Is_Error'] = is_error

        # Block info
        block_row = triallist['BLOCK_ROW']
        block_row = block_row[0, 0][0, 0]
        block_run_count = triallist['BLOCK_RUN_COUNT']
        block_run_count = block_run_count[0, 0][0, 0]

        # correct block count
        if part == 'part2':
            df['Block_Run_Count'] = block_run_count + 31
        elif part == '_Cues':
            df['Block_Run_Count'] = block_run_count + 60
        elif part == 'ues_2':
            df['Block_Run_Count'] = block_run_count + 60

        # info what phase of the experiment we are in
        # clamp trials are 'adaptation'
        if tp > 10 and tp < 20 and part == 'part1':
            df['Phase'] = 'Baseline'
        elif tp > 10 and tp < 20 and part == 'part2':
            df['Phase'] = 'Washout'
        elif part != 'part1' and part != 'part2':
            df['Phase'] = 'Generalization'
        else:
            df['Phase'] = 'Adaptation'

        # distance
        if tp > 30:
            df['Distance'] = 3
        elif tp > 20 and tp < 30:
            df['Distance'] = 1
        else:
            df['Distance'] = 2

        # force field direction, 0 = + 135°, 1 = -135°
        while tp > 10:
            tp = tp - 10

        df['Cue_Target'] = tp

        # clamp trial?
        if tp == 21 or tp == 22:
            df['Clamp_Trial'] = 1
        else:
            df['Clamp_Trial'] = 0

        # find out timing
        trial_abort = df.loc[df['Event_Label'] == 'TRIAL_ABORT', 'Milliseconds'].min()
        cue_light_on = df.loc[df['Event_Label'] == 'CUE_LIGHT_ON', 'Milliseconds'].min()
        cue_left = df.loc[df['Event_Label'] == 'CUE_LEFT', 'Milliseconds'].min()

        middle_light_on = df.loc[df['Event_Label'] == 'MIDDLE_LIGHT_ON', 'Milliseconds'].min()
        if pd.isna(middle_light_on):
            middle_light_on = df.loc[df['Event_Label'] == 'MIDDLE_LIGHT_ON_ALREADY_LEFT', 'Milliseconds'].min()

        reached_middle = df.loc[df['Event_Label'] == 'REACHED_MIDDLE', 'Milliseconds'].min()
        leave_middle = df.loc[df['Event_Label'] == 'LEAVE_MIDDLE', 'Milliseconds'].min()

        if pd.isna(trial_abort):  # normal trial
            dwell_time = leave_middle - reached_middle
            if pd.isna(cue_left):
                reaction_time = middle_light_on - cue_light_on
            else:
                reaction_time = cue_left - cue_light_on
        else:  # participants reached the middle target but it was aborted
            dwell_time = np.NaN
            reaction_time = np.NaN

        df['Reaction_Time'] = reaction_time
        df['Dwell_Time'] = dwell_time

        # adding trail to dataframe
        appended_df.append(df)

    # merging all trials into one dataframe
    df = pd.concat(appended_df, axis=0)
    print('Sample Points: ', len(df))

    # only keep Gernalization
    df = df.loc[(df['Phase'] == 'Generalization'), :]

    # checking dataframe
    df

    # %%
    # delete left hand variables from df: all columns which start with left*, + probably one
    df_right = df.drop(df.loc[:, df.columns.str.startswith('Left')], 1)

    # delete dataframes which were aborted
    df_fail = df_right.loc[df_right['Is_Error'] != 0, :]
    df_success = df_right.loc[df_right['Is_Error'] == 0, :]

    df_reach = df_right

    # %% save csv
    # one row per trial
    df_save = df_reach.groupby('Trial').mean()

    df_save.to_csv(r'/Users/mg/Desktop/MOVE/MOVE_Analysis_MG/KINARM/Generalization_Blocks/' +
                   vp + group + version + 'Generalization.csv')
