#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory plots of unimanual groups (Fig. 2A)

half of all trajectories of specific blocks are plotted.
(for each final target cue combination one trajectory per block per participant)

"""

from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %%
data_folder = op.join(config.paths['kinarm'], 'XY_trial/')
folder = listdir(data_folder)
folder

df_append = []
for file in folder:
    df = pd.read_csv(data_folder + file)
    df['VP'] = file[0:4]
    df['Group'] = file[4]
    df['Version'] = file[5]
    df_append.append(df)

df = pd.concat(df_append, axis=0)

# %% exclude participants who used explicit strategy

df = df.loc[df.VP != '0004', :]
df = df.loc[df.VP != '0007', :]
df = df.loc[df.VP != '0036', :]
df = df.loc[df.VP != '0039', :]
df = df.loc[df.VP != '0047', :]

# exclude clamp trials
df = df.loc[df.Clamp_Trial == 0, :]

# %% only keep half of the trials in one block
# (due to reviewer comment that one cannot distinguish between within and between variability)

block_liste = [6, 7, 56, 57]
Cue_Target_liste = [1, 2, 3, 4, 5, 6, 7, 8]

df_append = []
for vp in df.VP.unique():
    print(vp)
    for b in block_liste:
        print(b)
        for i in Cue_Target_liste:
            idx = ((df.VP == vp) &
                   (df.Block_Run_Count == b) &
                   (df.Cue_Target == i))
            t = df.loc[idx, 'Trial'].unique().min()
            idx = ((df.VP == vp) &
                   (df.Block_Run_Count == b) &
                   (df.Cue_Target == i) &
                   (df.Trial == t))
            df_trial = df.loc[idx, :]
            df_append.append(df_trial)

df_half = pd.concat(df_append, axis=0)

df = df_half

# %%
# split up df accorind to group
active = df.loc[df.Group == 'A', :]
control = df.loc[df.Group == 'C', :]
MI = df.loc[df.Group == 'M', :]

# split up according to Phase
active_base = active.loc[active.Block_Run_Count == 6, :]
active_adbe = active.loc[active.Block_Run_Count == 7, :]
active_adfi = active.loc[active.Block_Run_Count == 56, :]
active_wash = active.loc[active.Block_Run_Count == 57, :]

# hack for #56
control_base = control.loc[(control.Block_Run_Count == 6) &
                           (control.Phase == 'Baseline'), :]
control_adbe = control.loc[control.Block_Run_Count == 7, :]
control_adfi = control.loc[control.Block_Run_Count == 56, :]
control_wash = control.loc[control.Block_Run_Count == 57, :]

MI_base = MI.loc[MI.Block_Run_Count == 6, :]
MI_adbe = MI.loc[MI.Block_Run_Count == 7, :]
MI_adfi = MI.loc[MI.Block_Run_Count == 56, :]
MI_wash = MI.loc[MI.Block_Run_Count == 57, :]


# %% plot active group
active_group = [active_base, active_adbe, active_adfi, active_wash]
plot_names = ['baseline', 'adaptation_first', 'adaptation_last', 'washout']

for i, plot_data in enumerate(active_group):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x='X', y='Y', hue='Force_Direction',
                    palette=['crimson', 'black'], alpha=0.1, linewidth=3,
                    edgecolor=None, s=2,
                    data=plot_data, legend=False)
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set(yticklabels=[], xticklabels=[], xlabel='', ylabel='')
    fig.savefig(op.join(config.paths['plots'], f'half_active1_{plot_names[i]}.png'), dpi=300)

# %% plot control group
control_group = [control_base, control_adbe, control_adfi, control_wash]

for i, plot_data in enumerate(control_group):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x='X', y='Y', hue='Force_Direction',
                    palette=['darkkhaki', 'black'], alpha=0.1, linewidth=3,
                    edgecolor=None, s=2,
                    data=plot_data, legend=False)
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set(yticklabels=[], xticklabels=[], xlabel='', ylabel='')
    fig.savefig(op.join(config.paths['plots'], f'half_control_{plot_names[i]}.png'), dpi=300)

# %% plot MI group
MI_group = [MI_base, MI_adbe, MI_adfi, MI_wash]

for i, plot_data in enumerate(MI_group):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x='X', y='Y', hue='Force_Direction',
                    palette=['darkorange', 'black'], alpha=0.1, linewidth=3,
                    edgecolor=None, s=2,
                    data=plot_data, legend=False)
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set(yticklabels=[], xticklabels=[], xlabel='', ylabel='')
    fig.savefig(op.join(config.paths['plots'], f'half_MI_{plot_names[i]}.png'), dpi=300)
