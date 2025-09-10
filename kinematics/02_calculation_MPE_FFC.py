# -*- coding: utf-8 -*-
"""
*script from 2021*

pre-processing of data. calculation of MPE and FFC for unimanual groups.

Prerequisite: mat files are already created with matlab,
(the data is filtered and kinematic data is added)

- MPE is calculated as signed difference!
    (if MPE is in opposite direction of force field * -1)
- MPE is only calculated in the initial reach, so until the end of target

In this script I create and save csv files for each participant:
        1.) trajectories csv
            in XY_trial/{vp}{group}{version}MPEclean_trial.csv
            (to create trajectories plots)
        2.) csv file MPE averaged per block
            in MPE_Block/{vp}{group}{version}MPEclean.csv
            (to create lineplot)
        3.) csv file MPE every trial
            in MPE_trial/{vp}{group}{version}MPEclean_trial.csv
            (includes clamp trials so order/position of trials is correct)
        4.) every clamp trial value
            in CLAMP_Trial/{vp}{group}{version}.csv
        5.) clamp trial adaptation averaged over block
            in CLAMP_Block/{vp}{group}{version}.csv
            (to create lineplot)

- subject 0136 has their own script!

"""

# %%
from os import listdir
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import fnmatch

# %%
data_folder = '/data/pt_02349/MOVE/'
folder = listdir(data_folder)
folder

final_folder = '/data/p_02349/MOVE_Analysis_MG/KINARM/MPE_Block_new/'
final_list = listdir(final_folder)

for p in range(len(folder)):
    current_folder = folder[p]
    vp = folder[p][-4:]

    # C3D Import
    # define kinarm data path with unzipped .mat files
    subject_path = '/data/pt_02349/MOVE/' + current_folder + '/KINARM/'
    files = listdir(subject_path)

    vpk = vp + '*' + 'MPEclean.csv'

    # vp 136 has to be done separately because the mat files were to big to
    # join and save in matlab
    # execute only if file does not exist yet
    if fnmatch.filter(final_list, vpk) or vp == '0136':
        print(vpk)

    else:
        print('todo ' + vpk)

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

            # make a data frame, setting the time stamps as the index
            df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)

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

            # keep only data after middle is left
            df = df.loc[(df['Event_Label'] == 'LEAVE_MIDDLE') |
                        (df['Event_Label'] == 'MIDDLE_LIGHT_ON_ALREADY_LEFT'), :]

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

            # force field direction, 0 = + 135°, 1 = -135°
            if tp > 10:
                df['Cue_Target'] = tp - 10
            else:
                df['Cue_Target'] = tp

            # clamp trial?
            if tp == 21 or tp == 22:
                df['Clamp_Trial'] = 1
            else:
                df['Clamp_Trial'] = 0

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
            if tp in [1, 2, 11, 12, 5, 6, 15, 16]:

                df_out = df.loc[(df.Y < -12.625) |
                                (df.Y > 12.625), :]
                if not df_out.empty:
                    idx = df_out.index[0]
                    df_in = df.loc[:idx]
                elif df.empty:
                    df_in = df
                else:
                    idx = df.index[-1]
                    df_in = df.iloc[:idx]
                # now we have the index of the first time value was bigger than cutoff

                abs_X = abs(df_in.X)
                df['Deviation'] = abs_X
                if abs_X.empty:
                    df['Sign_Deviation'] = 0
                    max_abs = 0
                    sign = 0
                else:
                    max_abs = max(abs_X)
                    idx_max_abs_X = np.where(np.abs(df.X) == max_abs)[0][0]
                    sign = np.sign(df.X[df.index[idx_max_abs_X]])
                    df['Sign_Deviation'] = sign

            else:
                df_out = df.loc[(df.X < -12.625) |
                                (df.X > 12.625), :]
                if not df_out.empty:
                    idx = df_out.index[0]
                    df_in = df.loc[:idx]
                elif df.empty:
                    df_in = df
                else:
                    idx = df.index[-1]
                    df_in = df.iloc[:idx]
                # now we have the index of the first time value was bigger than cutoff

                abs_Y = abs(df_in.Y)
                df['Deviation'] = abs_Y
                if abs_Y.empty:
                    df['Sign_Deviation'] = 0
                    max_abs = 0
                    sign = 0
                else:
                    max_abs = max(abs_Y)
                    idx_max_abs_Y = np.where(np.abs(df.Y) == max_abs)[0][0]
                    sign = np.sign(df.Y[df.index[idx_max_abs_Y]])
                    df['Sign_Deviation'] = sign

            # force field direction
            if version == '1':
                if tp in [2, 3, 5, 7, 12, 13, 15, 17]:
                    df['Force_Direction'] = 0
                else:
                    df['Force_Direction'] = 1
                if tp in [1, 4, 5, 7, 11, 14, 15, 17]:
                    df['MPE_new'] = max_abs * sign * -1
                else:
                    df['MPE_new'] = max_abs * sign
            else:
                if tp in [2, 3, 5, 7, 12, 13, 15, 17]:
                    df['Force_Direction'] = 1
                else:
                    df['Force_Direction'] = 0
                if tp in [1, 4, 5, 7, 11, 14, 15, 17]:
                    df['MPE_new'] = max_abs * sign
                else:
                    df['MPE_new'] = max_abs * sign * -1

            # adding trail to dataframe
            appended_df.append(df)

        # merging all trials into one dataframe
        df = pd.concat(appended_df, axis=0)
        print('Sample Points: ', len(df))

        # checking dataframe
        df

        # %%
        # delete left hand variables from df: all columns which start with left*
        df_right = df.drop(df.loc[:, df.columns.str.startswith('Left')], 1)

        # delete dataframes which were aborted
        df_fail = df_right.loc[df_right['Is_Error'] != 0, :]
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
        df_reach = df_success

        # get clamp trials in separate df
        df_clamp = df_reach.loc[df_reach['Clamp_Trial'] == 1, :]
        # exclude trials from generalization phase
        df_clamp = df_clamp.loc[df_clamp['Phase'] != 'Generalization', :]

        df_reach = df_reach.loc[df_reach['Clamp_Trial'] == 0, :]

        # this count also excludes clamp trials
        trial_count = df_reach['Trial']
        new_trial_count = []
        counter = 0
        j = 0
        for i in trial_count:
            if i == j:
                new_trial_count.append(counter)
            else:
                counter = counter + 1
                new_trial_count.append(counter)
            j = i
        df_reach['Trial_Count'] = new_trial_count

        # %% exclude trials which look weird
        # delete trial which should be excluded - checked with exclusion_trials_check.py
        # I decided to exclude 7 trials
        # in blocks 6,7,55,56,57
        if vp == '0006':
            df_reach = df_reach.loc[df_reach.Trial != 1166, :]
        if vp == '0018':
            df_reach = df_reach.loc[df_reach.Trial != 1027, :]
        if vp == '0003':
            df_reach = df_reach.loc[df_reach.Trial != 1028, :]
        if vp == '0038':
            df_reach = df_reach.loc[df_reach.Trial != 1067, :]
        # in blocks used for mixed model (first + last 10 adaptation)
        if vp == '0057':
            df_reach = df_reach.loc[(df_reach.Trial != 88) &
                                    (df_reach.Trial != 290) &
                                    (df_reach.Trial != 976) &
                                    (df_reach.Trial != 1034), :]
        if vp == '0034':
            df_reach = df_reach.loc[df_reach.Trial != 895, :]

        # %% save csv file for trajectory plots
        df_reach.to_csv(r'/data/p_02349/MOVE_Analysis_MG/KINARM/XY_trial/' +
                        vp + group + version + 'MPEclean_trial.csv')

        # %% save csv block averaged MPE
        # exclude Generalization trials
        df_select = df_reach.loc[df_reach['Phase'] != 'Generalization', :]

        # calculate new MPE (signed values)
        df_max = df_select.groupby('Trial_Count').max()

        df_MPE = df_max.groupby('Block_Run_Count').Deviation.mean()
        df_MPE_new = df_max.groupby('Block_Run_Count').MPE_new.mean()

        df_MPE_new.to_csv(r'/data/p_02349/MOVE_Analysis_MG/KINARM/MPE_Block_new/' +
                          vp + group + version + 'MPEclean.csv')

        # %%
        # save MPE of every trial - including clamp trials
        df_max_2 = df_success.groupby('Trial_Count').max()

        df_MPE_trial = df_max_2[['MPE_new', 'Clamp_Trial', 'TP',
                                 'Cue_Target', 'Block_Run_Count', 'Phase']]

        df_MPE_trial.to_csv(r'/data/p_02349/MOVE_Analysis_MG/KINARM/MPE_trial/' +
                            vp + group + version + 'MPEclean_trial.csv')

        # %% compute FFC value
        # Extract data of 150ms window centered on the time of peak velocity
        # (only using vel in X direction):
        # Velocity profile -> force field force in normal trials
        # Linear regression with zero intercept:
        # Measured forces against channel walls ~ ideal force profile
        # Slope * 100% = adaptation performance
        # 100% = full adaptation
        # 0% = no adaptation

        # select trials where forces were measured during the clamp trials
        # first discard trajectory paths which happen after the target
        df_clamp2 = df_clamp.loc[df_clamp.X > -12, :]

        # exclude trials where recorded max force is close to 0
        df_clamp2['Force_Y'] = abs(df_clamp2.Right_Hand_ForceCMD_Y)
        # get series with true/false Force < 0.1
        tf_ser = df_clamp2.groupby('Trial').Force_Y.max() < 0.1
        # get index (= trial numbers) where max force is < 0.1
        i_ser = tf_ser.loc[tf_ser.values == True].index
        # transform into array
        i_arr = np.array(i_ser)
        # df exluding trials where no force was generated
        df_clamp_clean = df_clamp[~df_clamp['Trial'].isin(i_arr)]
        # number of 'missing-force' trials
        nr_failed_clamp = len(i_arr)

        # calculate force field force which would have been present in normal trials
        df_clamp_clean['Ideal_Force'] = df_clamp_clean['Right_HandXVel'] * 13

        # based on TP Force Field would have been in the opposite direction
        if version == '1':
            df_clamp_clean.loc[df_clamp_clean['TP'] == 21, 'Ideal_Force'] = df_clamp_clean.loc[df_clamp_clean['TP'] == 21, 'Ideal_Force'] * (-1)
        else:
            df_clamp_clean.loc[df_clamp_clean['TP'] == 22, 'Ideal_Force'] = df_clamp_clean.loc[df_clamp_clean['TP'] == 22, 'Ideal_Force'] * (-1)

        df_151_appended = []

        # 151 ms window -> 75ms before and after
        for trial in df_clamp_clean['Trial'].unique():
            # select single trial df
            df_st = df_clamp_clean.loc[df_clamp_clean['Trial'] == trial, :].copy()
            # max_vel = df_st.Right_HandXVel.min()
            idx_max_vel = df_st.Right_HandXVel.idxmin()
            df_st_151 = df_st.loc[(idx_max_vel - 75): (idx_max_vel + 76)].copy()

            # add column with slope from regression
            reg_x = df_st_151['Ideal_Force']
            reg_x = np.array(reg_x)
            reg_x = np.reshape(reg_x, (-1, 1))

            reg_y = df_st_151['Right_Hand_ForceCMD_Y']
            reg_y = np.array(reg_y)

            reg = LinearRegression(fit_intercept=False).fit(reg_x, reg_y)

            slope = reg.coef_[0]
            df_st_151.loc[:, 'Slope'] = slope * 100

            # adding trail to dataframe
            df_151_appended.append(df_st_151)

        # merging all trials into one dataframe
        df_151 = pd.concat(df_151_appended, axis=0)
        print('Sample Points: ', len(df_151))

        # save clamp value per trial (so I can see how many values are missing)
        df_151_trial = df_151.groupby('Trial_Count').max()
        df_clamp_trial = df_151_trial[['Block_Run_Count', 'Slope']]
        df_clamp_trial.to_csv(r'/data/p_02349/MOVE_Analysis_MG/KINARM/CLAMP_Trial/' +
                              vp + group + version + '.csv')

        # %% save csv block averaged adaptation during clamp trials
        df_CLAMP = df_151_trial.groupby('Block_Run_Count').Slope.mean()

        df_CLAMP.to_csv(r'/data/p_02349/MOVE_Analysis_MG/KINARM/CLAMP_Block/' +
                        vp + group + version + '.csv')
