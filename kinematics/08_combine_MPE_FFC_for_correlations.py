#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine 3 kinematic measures into one measure "Behavior" and save csv

each measure is normalized first (divided by standard deviation) and then they
are summed up

"""

import matplotlib as mpl
mpl.use('TkAgg')
from os import listdir
import numpy as np
import pandas as pd
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %%
data_folder = op.join(config.paths['kinarm'], 'MPE_Block_new/')
folder = listdir(data_folder)
folder

df1_append = []
for file in folder:
    #print(file)
    df1 = pd.read_csv(data_folder+file)
    df1['VP'] = file[0:4]
    df1['Group'] = file[4]
    df1['Version'] = file[5]
    df1['Phase'] = 2
    df1.loc[df1.Block_Run_Count < 7, 'Phase'] = 1
    df1.loc[df1.Block_Run_Count > 56, 'Phase'] = 3

    df1_append.append(df1)

df1 = pd.concat(df1_append, axis=0)

# %% exclude participants #
df1 = df1.loc[df1.VP != '0004', :]
df1 = df1.loc[df1.VP != '0007', :]
df1 = df1.loc[df1.VP != '0036', :]
df1 = df1.loc[df1.VP != '0039', :]
df1 = df1.loc[df1.VP != '0047', :]


# %% adaptation phase
df_adapt = df1.loc[df1.Phase == 2, :]

# get first block
df_adapt_7 = df_adapt.loc[df_adapt.Block_Run_Count == 7, :]
# get last 2 blocks
df_adapt_4950 = df_adapt.loc[(df_adapt.Block_Run_Count == 55) |
                             (df_adapt.Block_Run_Count == 56), :]

# average last 2 blocks
adapt_555 = df_adapt_4950.groupby('VP', as_index = False).mean()


# sort according to participant number
df_adapt_7.sort_values(by=['VP'], inplace=True)

# merge 2 dataframes
df = pd.merge(df_adapt_7, adapt_555, on = 'VP', how = 'inner')

# MPE_new_x is first block adapt
# MPE_new_y is avergae of last 2 blocks adapt


# %% calculation of difference + df clean up
df['MPE_adaptation_change'] = df['MPE_new_y'] - df['MPE_new_x']

df = df.drop(['Block_Run_Count_x', 'Block_Run_Count_y', 'Phase_x','Phase_y', 'MPE_new_x', 'MPE_new_y'], axis=1)

# %% baseline washout MPE
df_no_b = df1.loc[df1.Block_Run_Count == 6, :]
df_no_w = df1.loc[df1.Block_Run_Count == 57, :]

# merge 2 dataframes
df_no = pd.merge(df_no_b, df_no_w, on = 'VP', how = 'inner')

# differences washout - baseline
df_no['MPE_basewash_change'] = df_no['MPE_new_y'] - df_no['MPE_new_x']
df_no = df_no.drop(['Block_Run_Count_x','Block_Run_Count_y', 'MPE_new_x','MPE_new_y', 'Version_x', 'Version_y', 'Phase_x', 'Phase_y', 'Group_x', 'Group_y'], axis=1)

# add baseline washout change to df
df = pd.merge(df, df_no, on='VP', how='inner')


# %% load clamp trials data
data_folder = op.join(config.paths['kinarm'], 'CLAMP_Block/')
folder = listdir(data_folder)
folder

df1_append = []
for file in folder:
    #print(file)
    df1 = pd.read_csv(data_folder+file)
    df1['VP'] = file[0:4]
    df1['Group'] = file[4]
    df1['Version'] = file[5]
    df1['Phase'] = 2
    df1.loc[df1.Block_Run_Count < 7, 'Phase'] = 1
    df1.loc[df1.Block_Run_Count > 56, 'Phase'] = 3

    df1_append.append(df1)

df1 = pd.concat(df1_append, axis=0)

# %% exclude participants #
df1 = df1.loc[df1.VP != '0004', :]
df1 = df1.loc[df1.VP != '0007', :]
df1 = df1.loc[df1.VP != '0036', :]
df1 = df1.loc[df1.VP != '0039', :]
df1 = df1.loc[df1.VP != '0047', :]


# %% adaptation phase

df_adapt = df1.loc[(df1.Block_Run_Count == 53) |
                  (df1.Block_Run_Count == 54) |
                  (df1.Block_Run_Count == 55) |
                  (df1.Block_Run_Count == 56), :]

# average last 2 blocks
adapt_cl = df_adapt.groupby(['Group', 'VP'], as_index = False).mean()

adapt_cl = adapt_cl.drop(['Group', 'Block_Run_Count', 'Phase'], axis=1)

df = pd.merge(df, adapt_cl, on='VP', how='inner')


# %% normalize measures

df['MPE_adaptation_norm'] = (df.MPE_adaptation_change) / np.std(df.MPE_adaptation_change)
df['MPE_basewash_norm'] = (df.MPE_basewash_change) / np.std(df.MPE_basewash_change)
df['Slope_norm'] = (df.Slope) / np.std(df.Slope)

df['Behavior'] = df.MPE_adaptation_norm + df.MPE_basewash_norm + (df.Slope_norm * (-1))

df.MPE_basewash_change.corr(df.Behavior, method = 'pearson')

#x = df.Slope
#y = df.Behavior
#plt.scatter(x=x, y=y)

df.to_csv(op.join(config.paths['kinarm'], 'behavioral_measure.csv'))


