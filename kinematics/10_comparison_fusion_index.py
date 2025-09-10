#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison of fusion index between groups

in task 3 all participants made active reaches:
2 blocks * 24 trials = 8 trials * 3 distances * 2 blocks

creates supplementary fig. S1

save median values for correlation
"""
# %%
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind

import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %%
data_folder = op.join(config.paths['kinarm'], 'fusion_index/')
folder = listdir(data_folder)
folder.remove('.DS_Store')

df_append = []
for file in folder:
    df = pd.read_csv(data_folder + file)
    df['VP'] = file[-8:-4]
    df_append.append(df)

df = pd.concat(df_append, axis=0)

df = df.rename(columns={'Unnamed: 0': 'Trial'})
df = df.sort_values(['VP', 'Trial'])

# exclude participants
df = df.loc[df.VP != '0004', :]
df = df.loc[df.VP != '0007', :]
df = df.loc[df.VP != '0036', :]
df = df.loc[df.VP != '0039', :]
df = df.loc[df.VP != '0047', :]

# %% create df for plotting
df.loc[df.Group == 'C', 'Group'] = 1
df.loc[df.Group == 'M', 'Group'] = 2
df.loc[df.Group == 'A', 'Group'] = 3

df_median = df.groupby(df.VP).median(numeric_only=False)

# %%
df_control = df.loc[df.Group == 1, :]
df_MI = df.loc[df.Group == 2, :]
df_active = df.loc[df.Group == 3, :]

# %%
reaction_control = []
for vp in df_control.VP.unique():
    mean = (df_control.loc[(df_control.VP == vp), 'Fusion_Index']).median()
    reaction_control.append(mean)

reaction_MI = []
for vp in df_MI.VP.unique():
    mean = (df_MI.loc[(df_MI.VP == vp), 'Fusion_Index']).median()
    reaction_MI.append(mean)

reaction_active = []
for vp in df_active.VP.unique():
    mean = (df_active.loc[(df_active.VP == vp), 'Fusion_Index']).median()
    reaction_active.append(mean)

t, p1 = ttest_ind(reaction_MI, reaction_control, permutations=1000000)
t, p2 = ttest_ind(reaction_MI, reaction_active, permutations=1000000)
t, p3 = ttest_ind(reaction_control, reaction_active, permutations=1000000)

median_control = np.median(reaction_control)
median_MI = np.median(reaction_MI)
median_active = np.median(reaction_active)

# %%
Y = 'Fusion_Index'

fig, ax = plt.subplots(figsize=(5, 8))

palette = ['darkkhaki', 'darkorange', 'crimson']

# Create violin plots without mini-boxplots inside.
ax = sns.violinplot(y=Y, x='Group', data=df_median,
                    hue='Group', palette=palette, linewidth=0,
                    cut=0.7, inner=None, dodge=False, order=[1, 2, 3]) # cut = 0.7

# Clip the right half of each violin.
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                       transform=ax.transData))

num_items = len(ax.collections)
sns.stripplot(y=Y, x='Group', hue='Group', palette=palette,
              edgecolor='white', linewidth=2, jitter=0.15,
              data=df_median, s=10, order=[1, 2, 3], alpha=0.6) #  x='Group',


sns.boxplot(y=Y, x='Group', data=df_median, width=0.2,
            showfliers=False, showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='black',
                           markersize=12, zorder=3, markeredgecolor='white'),
            boxprops=dict(facecolor=(0, 0, 0, 0),
                          linewidth=2.5, zorder=3, edgecolor='black'),
            whiskerprops=dict(linewidth=2.5, color='black'),
            capprops=dict(linewidth=2.5, color='black'),
            medianprops=dict(linewidth=2.5, color='black'), order=[1, 2, 3])


ax.set_ylim((0, 1.1))
ax.set_xticklabels([' ', ' ', ' '])
ax.tick_params(axis='x', which='both', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('')
plt.ylabel('Fusion Index')

handle1 = mpatches.Patch(color=palette[0], label='control')
handle2 = mpatches.Patch(color=palette[1], label='motor imagery')
handle3 = mpatches.Patch(color=palette[2], label='active')


plt.legend(handles=[handle1, handle2, handle3],
           title='Group',
           loc='upper left',
           facecolor='White',
           bbox_to_anchor=(0, 1),
           fancybox=True,
           edgecolor='white',
           framealpha=1)

fig.savefig(op.join(config.paths['plots'], 'Fusion_Index.svg'))

# %% save csv
df_median.loc[df_median.Group == 1, 'Group'] = 'C'
df_median.loc[df_median.Group == 2, 'Group'] = 'M'
df_median.loc[df_median.Group == 3, 'Group'] = 'A'

df_median = df_median.drop(['Trial'], axis=1)

df_median.to_csv(op.join(config.paths['kinarm'], 'fusion_index.csv'))
