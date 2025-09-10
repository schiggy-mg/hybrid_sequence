# -*- coding: utf-8 -*-
"""
pre-processing of data. calculation of fusion index for timing check task.

annoyingly I have to get the data from the mat files (could be integrated in
script 02)

Prerequisite: mat files are already created with matlab,
(the data is filtered and kinematic data is added)

In this script I create and save csv files for each participant:
        1.) fusion index csv

"""

# %%
from os import listdir
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# %% 
data_folder = '/Volumes/MOVE_data/MOVE'
folder = listdir(data_folder)

for p in range(len(folder)):
    print(p)
    current_folder = folder[p]
    vp = folder[p][-4:]
    print(vp)
    if (vp != 'tore'):

        # C3D Import
        # define kinarm data path with unzipped .mat files
        subject_path = f'{data_folder}/{current_folder}/KINARM/'
        files = listdir(subject_path)

        print('todo ' + vp)

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

        # %%

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

            # keep only data of reaches
            df = df.loc[(df['Event_Label'] != 'HAND_IN_SPACE') &
                        (df['Event_Label'] != 'START_MOVING_TO_CUE') &
                        (df['Event_Label'] != 'REACHED_CUE') &
                        (df['Event_Label'] != 'CUE_LIGHT_ON') &
                        (df['Event_Label'] != 'SHOW_FIX') &
                        (df['Event_Label'] != 'PAUSE') &
                        (df['Event_Label'] != 'REACHED_GOAL'), :]


            # Initialize variables
            c_found = False

            # Create the new column X, indicate which reach we are on
            # reached_middle is part of the second reach because if the middle is
            # left before the light, we do not have a LEAVE_MIDDLE trigger
            df['Reach'] = 1

            # Iterate through the DataFrame
            for i in df.index:
                if df.loc[i, 'Event_Label'] == 'REACHED_MIDDLE':
                    c_found = True
                if c_found:
                    df.loc[i, 'Reach'] = 2


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

            # vp 56 recalibration during task 1
            if (part == 'part1' and vp == '0056' and trial > 423):
                df['Block_Run_Count'] = block_run_count + 23
                switch = 1
            else:
                df['Block_Run_Count'] = block_run_count

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

            # keep only Generalization trials
            df = df.loc[(df['Phase'] == 'Generalization'), :]

            # deviation from straight reach depends on tp
            # create X and Y coordinates, where 0/0 is the middle position
            # for this we need to correct by a constants which are different for each participant
            if part == 'part1':
                df['X'] = (df['Right_HandX'] - x_offset) * 100
                df['Y'] = (df['Right_HandY'] - y_offset) * 100
            else:
                df['X'] = (df['Right_HandX'] - x_offset2) * 100
                df['Y'] = (df['Right_HandY'] - y_offset2) * 100

            # Sign_Deviation = deviation is positive or negative
            # exclude trajectories when participants have reached too far
            #if tp in [1, 2, 11, 12, 5, 6, 15, 16]:
            if tp in [1, 2, 11, 12, 5, 6, 15, 16, 21, 22, 25, 26, 31, 32, 35, 36]:

                df = df.loc[(df.Y > -12.625) |
                                (df.Y < 12.625), :]


            else:
                df = df.loc[(df.X > -12.625) |
                                (df.X < 12.625), :]

            # adding trail to dataframe
            appended_df.append(df)

        # merging all trials into one dataframe
        df = pd.concat(appended_df, axis=0)
        print('Sample Points: ', len(df))


        # %%
        # delete left hand variables from df: all columns which start with left*, + probably one
        df_right = df.drop(df.loc[:, df.columns.str.startswith('Left')], axis=1)

        # delete dataframes which were aborted
        #df_fail = df_right.loc[df_right['Is_Error'] != 0, :]
        df_success = df_right.loc[df_right['Is_Error'] == 0, :]

        # new trial count excluding aborted trials
        trial_count = df_success['Trial']
        new_trial_count = []
        counter = 0
        j = 1
        for i in trial_count:
            if i == j:
                new_trial_count.append(counter)
            else:
                counter = counter + 1
                new_trial_count.append(counter)
            j = i

        df_success['Trial_Count'] = new_trial_count

        # %%
        df_success['Velocity'] = np.sqrt((df_success['Right_HandXVel'] ** 2) +
                                        (df_success['Right_HandYVel'] ** 2))

        # Define index array
        trials = df_success['Trial_Count'].unique()

        # Pre-allocate the DataFrame with the specified index and an additional column
        df_FI = pd.DataFrame(index=trials, columns=['Fusion_Index', 'Group', 'VP'])

        df_FI.Group = group
        df_FI.VP = vp

        for trial in df_success['Trial_Count'].unique():

            reach_1 = pd.NA
            reach_2 = pd.NA
            switch_reach = pd.NA

            vel_curve = df_success.loc[df_success['Trial_Count'] == trial, 'Velocity']

            middle_reached = df_success.loc[(df_success['Event_Label'] == 'REACHED_MIDDLE') &
                                (df_success['Trial_Count'] == trial), 'Milliseconds'].values[0]
            x2 = df_success.loc[(df_success['Event_Label'] == 'REACHED_MIDDLE') &
                                (df_success['Trial_Count'] == trial), 'Milliseconds'].values[-1]
            middle_left = df_success.loc[(df_success['Event_Label'] == 'LEAVE_MIDDLE') &
                                (df_success['Trial_Count'] == trial), 'Milliseconds'].values[0]

            arr = vel_curve.values
            peak_arr_idx = argrelextrema(arr, np.greater)[0]  # index von maxima
            peak_arr = arr[peak_arr_idx]

            trough_arr_idx = argrelextrema(arr, np.less)[0]  # index von minima
            trough_arr = arr[trough_arr_idx]

            start_of_trial = df_success.loc[df_success['Trial_Count'] == trial, 'Milliseconds'].values[0]

            if trough_arr.size > 0:
                trough_arr_idx_ms = trough_arr_idx[0] + start_of_trial  # index of first minima in ms

            if peak_arr.size > 0:
                peak_arr_idx_ms = peak_arr_idx[0] + start_of_trial  # index of first minima in ms


            if peak_arr.shape[0] != 0:
                reach_1 = peak_arr[0]
            else:
                reach_1 = arr[-1]

            if peak_arr.shape[0] > 1:
                reach_2 = peak_arr[1]
            else:  # if only one peak use vel value at end
                reach_2 = arr[-1]

            if trough_arr.size > 0:  # if array not empty
                if trough_arr_idx_ms > (middle_left + 150):  # trough later than 150ms after middle left
                    switch_reach = pd.NA

                elif trough_arr_idx_ms < (middle_reached - 150):  # trough before 150ms before middle reached
                    if trough_arr.size > 1:
                        switch_reach = trough_arr[1]
                        if peak_arr_idx_ms < trough_arr_idx_ms:
                            reach_1 = peak_arr[1]
                            if peak_arr.shape[0] > 2:
                                reach_2 = peak_arr[2]
                            else:
                                reach_2 = arr[-1]
                    else:
                        switch_reach = pd.NA

                elif trough_arr.size > 1:  # more troughs than one
                    trough_arr_idx_ms_2 = trough_arr_idx[1] + start_of_trial  # index of second minima in ms
                    if (trough_arr_idx_ms_2 >= (middle_reached - 150)) & (trough_arr_idx_ms_2 < (middle_left + 150)):
                        switch_reach = np.min([trough_arr[0], trough_arr[1]])
                        if peak_arr.shape[0] > 2:
                            reach_2 = np.max([peak_arr[1], peak_arr[2]])
                        else:  # if only one peak use vel value at end
                            reach_2 = arr[-1]

                        if trough_arr.size > 2:
                            trough_arr_idx_ms_3 = trough_arr_idx[2] + start_of_trial  # index of second minima in ms
                            if (trough_arr_idx_ms_3 >= (middle_reached - 150)) & (trough_arr_idx_ms_3 < (middle_left + 150)):
                                switch_reach = np.min([switch_reach, trough_arr[2]])
                                if peak_arr.shape[0] > 2:
                                    reach_2 = np.max([reach_2, peak_arr[3]])
                                else:  # if only one peak use vel value at end
                                    reach_2 = arr[-1]


                            if trough_arr.size > 3:
                                trough_arr_idx_ms_4 = trough_arr_idx[3] + start_of_trial  # index of second minima in ms
                                if (trough_arr_idx_ms_4 >= (middle_reached - 150)) & (trough_arr_idx_ms_4 < (middle_left + 150)):
                                    switch_reach = np.min([switch_reach, trough_arr[3]])
                    else:
                        switch_reach = trough_arr[0]

                else:
                    switch_reach = trough_arr[0]

            else:  # no trough at all
                switch_reach = pd.NA

            # calculate fusion index
            FI = 1 - (((reach_1 + reach_2) / 2) - switch_reach) / ((reach_1 + reach_2) / 2)

            if pd.isna(switch_reach):
                FI = 1

            df_FI.loc[trial, 'Fusion_Index'] = FI

            df_FI.to_csv(f'/Users/mg/Desktop/MOVE/MOVE_Analysis_MG/KINARM/fusion_index/FI_{vp}.csv')
