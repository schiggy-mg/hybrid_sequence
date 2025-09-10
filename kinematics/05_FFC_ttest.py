#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checks how many relevant trials are missing

Comparison of FFC in the last four blocks. within and across participants.

creates plot of FFC group comparison (plot 2F)

calculates how different 400ms participants were from average of group!
(here only motor imagery group)

"""

from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import numpy as np
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %% find out how many trials are missing (bimanual groups, adjust code for unimanual groups)

control_analysis = 0  # 1 to check results if I look at FFc change from last four baseline to last four adaptation blocks

data_folder = op.join(config.paths['kinarm'], 'CLAMP_Trial/')
folder = listdir(data_folder)
folder

df_append = []
for file in folder:
    df = pd.read_csv(data_folder + file)
    df['VP'] = file[0:4]
    df['Group'] = file[4]
    df['Version'] = file[5]
    df['Phase'] = 2
    df.loc[df.Block_Run_Count < 7, 'Phase'] = 1
    df.loc[df.Block_Run_Count > 56, 'Phase'] = 3
    df.loc[df.Block_Run_Count > 60, 'Phase'] = 4
    df.loc[df.Block_Run_Count > 65, 'Phase'] = 5

    df_append.append(df)

df = pd.concat(df_append, axis=0)

# exclude participants #
df = df.loc[df.VP != '0004', :]
df = df.loc[df.VP != '0007', :]
df = df.loc[df.VP != '0036', :]
df = df.loc[df.VP != '0039', :]
df = df.loc[df.VP != '0047', :]

# only keep relevant blocks 53, 54, 55, 56
if control_analysis == 0:
    df = df.loc[(df.Block_Run_Count == 53) |
                (df.Block_Run_Count == 54) |
                (df.Block_Run_Count == 55) |
                (df.Block_Run_Count == 56), :]

else:
    df = df.loc[(df.Block_Run_Count == 3) |
                (df.Block_Run_Count == 4) |
                (df.Block_Run_Count == 5) |
                (df.Block_Run_Count == 6) |
                (df.Block_Run_Count == 53) |
                (df.Block_Run_Count == 54) |
                (df.Block_Run_Count == 55) |
                (df.Block_Run_Count == 56), :]


# in total:
# 73 tirlas missing out of 480 trials

# %% load csv files for real analysis

data_folder = op.join(config.paths['kinarm'], 'CLAMP_Block/')
folder = listdir(data_folder)
folder

df_append1 = []
for file in folder:
    df1 = pd.read_csv(data_folder + file)
    df1['VP'] = file[0:4]
    df1['Group'] = file[4]
    df1['Version'] = file[5]
    df1['Phase'] = 2
    df1.loc[df1.Block_Run_Count < 7, 'Phase'] = 1
    df1.loc[df1.Block_Run_Count > 56, 'Phase'] = 3
    df1.loc[df1.Block_Run_Count > 60, 'Phase'] = 4
    df1.loc[df1.Block_Run_Count > 65, 'Phase'] = 5

    df_append1.append(df1)

df1 = pd.concat(df_append1, axis=0)

# exclude participants #
df1 = df1.loc[df1.VP != '0004', :]
df1 = df1.loc[df1.VP != '0007', :]
df1 = df1.loc[df1.VP != '0036', :]
df1 = df1.loc[df1.VP != '0039', :]
df1 = df1.loc[df1.VP != '0047', :]

df1.reset_index(drop=True, inplace=True)

# %%
if control_analysis == 0:
    df_adapt1 = df1.loc[(df1.Block_Run_Count == 53) |
                        (df1.Block_Run_Count == 54) |
                        (df1.Block_Run_Count == 55) |
                        (df1.Block_Run_Count == 56), :]

    # average last 4 blocks
    adapt_cl1 = df_adapt1.groupby(['Group', 'VP'], as_index=False).mean()
else:
    df_baseline1 = df1.loc[(df1.Block_Run_Count == 3) |
                           (df1.Block_Run_Count == 4) |
                           (df1.Block_Run_Count == 5) |
                           (df1.Block_Run_Count == 6), :]
    df_adapt1 = df1.loc[(df1.Block_Run_Count == 53) |
                        (df1.Block_Run_Count == 54) |
                        (df1.Block_Run_Count == 55) |
                        (df1.Block_Run_Count == 56), :]
    baseline_mean1 = df_baseline1.groupby(['Group', 'VP'], as_index=False).mean()
    adapt_cl1 = df_adapt1.groupby(['Group', 'VP'], as_index=False).mean()
    adapt_cl1['baseline_Slope'] = baseline_mean1.Slope
    adapt_cl1['Slope'] = adapt_cl1.Slope - adapt_cl1.baseline_Slope

# %% t-tests
# between groups
ac = adapt_cl1.loc[adapt_cl1.Group == 'A', 'Slope']
mi = adapt_cl1.loc[adapt_cl1.Group == 'M', 'Slope']
co = adapt_cl1.loc[adapt_cl1.Group == 'C', 'Slope']

t_acmi, p_acmi = ttest_ind(ac, mi)
t_acco, p_acco = ttest_ind(ac, co)
t_mico, p_mico = ttest_ind(mi, co)

# within groups
zeros = [0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]

t_ac, p_ac = ttest_rel(ac, zeros)
t_mi, p_mi = ttest_rel(mi, zeros)
t_co, p_co = ttest_rel(co, zeros)

# bonferroni holm corrected values
# sort and correct: smallest value multiplied by n, second smallest by n-1 etc.
p_values = [p_ac, p_mi, p_co, p_acmi, p_acco, p_mico]
np.sort(p_values)

p_acco * 6
p_ac * 5
p_acmi * 4
p_mi * 3
p_mico * 2
p_co

# %%
fig, ax = plt.subplots(figsize=(5, 8))

palette = ['crimson', 'darkkhaki', 'darkorange']

# Create violin plots without mini-boxplots inside.
ax = sns.violinplot(y='Slope', x='Group', data=adapt_cl1,
                    hue='Group', palette=palette, linewidth=0,
                    cut=0.7, inner=None, dodge=False, order=['C', 'M', 'A'])

# Clip the right half of each violin.
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                       transform=ax.transData))

num_items = len(ax.collections)
sns.stripplot(y='Slope', x='Group', hue='Group', palette=palette,
              edgecolor='white', linewidth=2, jitter=0.15,
              data=adapt_cl1, s=10, order=['C', 'M', 'A'], alpha=0.6)

sns.boxplot(y='Slope', x='Group', data=adapt_cl1, width=0.2,
            showfliers=False, showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='black',
                           markersize=12, zorder=3, markeredgecolor='white'),
            boxprops=dict(facecolor=(0, 0, 0, 0),
                          linewidth=2.5, zorder=3, edgecolor='black'),
            whiskerprops=dict(linewidth=2.5, color='black'),
            capprops=dict(linewidth=2.5, color='black'),
            medianprops=dict(linewidth=2.5, color='black'), order=['C', 'M', 'A'])

plt.axhline(0, color='Black', linewidth=1, linestyle='--')

ax.set_ylim((-25, 100))
ax.set_xticklabels([' ', ' ', ' '])
ax.tick_params(axis='x', which='both', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('')
plt.ylabel('MPE change adaptation [cm]')

handle1 = mpatches.Patch(color=palette[1], label='control')
handle2 = mpatches.Patch(color=palette[2], label='motor imagery')
handle3 = mpatches.Patch(color=palette[0], label='active')

plt.legend(handles=[handle1, handle2, handle3],
           title='Group',
           loc='upper left',
           facecolor='White',
           bbox_to_anchor=(0.5, 0.18),
           fancybox=True,
           edgecolor='white',
           framealpha=1,
           )

fig.savefig(op.join(config.paths['plots'], 'FFC_reduc.svg'))

# %% check different timing

# only MI group here
active = adapt_cl1.loc[adapt_cl1.Group == 'M', ['VP', 'Slope']]  # 3, 9

active_400 = active.loc[(active.VP == '0003') |
                        (active.VP == '0009'), :]

active_600 = active.loc[(active.VP != '0003') &
                        (active.VP != '0009'), :]


mean_active_400 = np.mean(active_400.Slope)
mean_active_600 = np.mean(active_600.Slope)
std_active_600 = np.std(active_600.Slope)

active_difference = (mean_active_400 - mean_active_600) / std_active_600
