#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

comparison of MPE change between groups:

For each participant, the difference of the average MPE of the first
and last two adaptation blocks and the difference of the average
MPE of the last baseline and first washout block is calculated

MPE adaptation difference and MPE washout difference are compared within and
across groups.

creates plots 2C and 2D

calculates how different 400ms participants were from average of group!
(here only motor imagery group)

"""

from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %%
data_folder = op.join(config.paths['kinarm'], 'MPE_Block_new/')
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
df.reset_index(drop=True, inplace=True)

# exclude participants
df = df.loc[df.VP != '0004', :]
df = df.loc[df.VP != '0007', :]
df = df.loc[df.VP != '0036', :]
df = df.loc[df.VP != '0039', :]
df = df.loc[df.VP != '0047', :]

# %%
# quick plot of all participants
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x='Block_Run_Count', y='MPE_new', hue='Group',
             estimator='mean', ci=95, n_boot=100,
             palette=['red', 'blue', 'teal'],
             data=df, legend=True)

plt.hlines(0, -1, 67, colors='Black', linewidth=0.5)

# %% adaptation phase
df_adapt = df.loc[df.Phase == 2, :]

# get first block
df_adapt_7 = df_adapt.loc[df_adapt.Block_Run_Count == 7, :]
# get last 2 blocks
df_adapt_4950 = df_adapt.loc[(df_adapt.Block_Run_Count == 55) |
                             (df_adapt.Block_Run_Count == 56), :]

# average last 2 blocks
adapt_555 = df_adapt_4950.groupby('VP', as_index=False).mean()

# sort df by VP
df_adapt_7.sort_values(by=['VP'], inplace=True)

# merge 2 dataframes
stat_adapt = pd.merge(df_adapt_7, adapt_555, on='VP', how='inner')

# MPE_new_x is first block adapt
# MPE_new_y is avergae of last 2 blocks adapt

# %% calculation of difference + df clean up
stat_adapt['MPE_reduc'] = stat_adapt['MPE_new_y'] - stat_adapt['MPE_new_x']

stat_adapt = stat_adapt.drop(['Block_Run_Count_x', 'Block_Run_Count_y', 'Phase_x', 'Phase_y'], axis=1)
stat_adapt.rename(columns={'MPE_new_x': 'MPE_begin'}, inplace=True)
stat_adapt.rename(columns={'MPE_new_y': 'MPE_end'}, inplace=True)

# %% t tests
# between groups
ac = stat_adapt.loc[stat_adapt.Group == 'A', 'MPE_reduc']
mi = stat_adapt.loc[stat_adapt.Group == 'M', 'MPE_reduc']
co = stat_adapt.loc[stat_adapt.Group == 'C', 'MPE_reduc']

t_acmi, p_acmi = ttest_ind(ac, mi)
t_acco, p_acco = ttest_ind(ac, co)
t_mico, p_mico = ttest_ind(mi, co)

# within groups
ac_begin = stat_adapt.loc[stat_adapt.Group == 'A', 'MPE_begin']
mi_begin = stat_adapt.loc[stat_adapt.Group == 'M', 'MPE_begin']
co_begin = stat_adapt.loc[stat_adapt.Group == 'C', 'MPE_begin']
ac_end = stat_adapt.loc[stat_adapt.Group == 'A', 'MPE_end']
mi_end = stat_adapt.loc[stat_adapt.Group == 'M', 'MPE_end']
co_end = stat_adapt.loc[stat_adapt.Group == 'C', 'MPE_end']

t_ac, p_ac = ttest_rel(ac_begin, ac_end)
t_mi, p_mi = ttest_rel(mi_begin, mi_end)
t_co, p_co = ttest_rel(co_begin, co_end)

# bonferroni holm corrected values
# sort and correct: smallest value multiplied by n, second smallest by n-1 etc.
p_values = [p_ac, p_mi, p_co, p_acmi, p_acco, p_mico]
np.sort(p_values)

p_acco * 6
p_ac * 5
p_mi * 4
p_acmi * 3
p_mico * 2
p_co


# %% difference plots MPE adaptation

fig, ax = plt.subplots(figsize=(5, 8))

palette = ['darkkhaki', 'crimson', 'darkorange']

# Create violin plots without mini-boxplots inside.
ax = sns.violinplot(y='MPE_reduc', x='Group', data=stat_adapt,
                    hue='Group', palette=palette, linewidth=0,
                    cut=0.7, inner=None, dodge=False, order=['C', 'M', 'A'])

# Clip the right half of each violin.
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                       transform=ax.transData))

num_items = len(ax.collections)
sns.stripplot(y='MPE_reduc', x='Group', hue='Group', palette=palette,
              edgecolor='white', linewidth=2, jitter=0.15,
              data=stat_adapt, s=10, order=['C', 'M', 'A'], alpha=0.6)

sns.boxplot(y='MPE_reduc', x='Group', data=stat_adapt, width=0.2,
            showfliers=False, showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='black',
                           markersize=12, zorder=3, markeredgecolor='white'),
            boxprops=dict(facecolor=(0, 0, 0, 0),
                          linewidth=2.5, zorder=3, edgecolor='black'),
            whiskerprops=dict(linewidth=2.5, color='black'),
            capprops=dict(linewidth=2.5, color='black'),
            medianprops=dict(linewidth=2.5, color='black'), order=['C', 'M', 'A'])

plt.axhline(0, color='Black', linewidth=1, linestyle='--')

ax.set_ylim((-4, 1.5))
ax.set_xticklabels([' ', ' ', ' '])
ax.tick_params(axis='x', which='both', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('')
plt.ylabel('MPE change adaptation [cm]')

handle1 = mpatches.Patch(color=palette[0], label='control')
handle2 = mpatches.Patch(color=palette[2], label='motor imagery')
handle3 = mpatches.Patch(color=palette[1], label='active')

plt.legend(handles=[handle1, handle2, handle3],
           title='Group',
           loc='upper left',
           facecolor='White',
           bbox_to_anchor=(0.05, 0.18),
           fancybox=True,
           edgecolor='white',
           framealpha=1
           )

fig.savefig(op.join(config.paths['plots'], 'MPE_reduc_adaptation.svg'))

# %% MPE baseline and washout
df_no_b = df.loc[df.Block_Run_Count == 6, :]
df_no_w = df.loc[df.Block_Run_Count == 57, :]

# merge 2 dataframes
df_no = pd.merge(df_no_b, df_no_w, on='VP', how='inner')

df_no['MPE_reduc'] = df_no['MPE_new_y'] - df_no['MPE_new_x']

# %% t-tests washout effect
# between groups
ac = df_no.loc[df_no.Group_y == 'A', 'MPE_reduc']
mi = df_no.loc[df_no.Group_y == 'M', 'MPE_reduc']
co = df_no.loc[df_no.Group_y == 'C', 'MPE_reduc']

t_acmi, p_acmi = ttest_ind(ac, mi)
t_acco, p_acco = ttest_ind(ac, co)
t_mico, p_mico = ttest_ind(mi, co)

# within groups
ac_base = df_no.loc[df_no.Group_y == 'A', 'MPE_new_x']
mi_base = df_no.loc[df_no.Group_y == 'M', 'MPE_new_x']
co_base = df_no.loc[df_no.Group_y == 'C', 'MPE_new_x']
ac_wash = df_no.loc[df_no.Group_y == 'A', 'MPE_new_y']
mi_wash = df_no.loc[df_no.Group_y == 'M', 'MPE_new_y']
co_wash = df_no.loc[df_no.Group_y == 'C', 'MPE_new_y']

t_ac, p_ac = ttest_rel(ac_base, ac_wash)
t_mi, p_mi = ttest_rel(mi_base, mi_wash)
t_co, p_co = ttest_rel(co_base, co_wash)

# bonferroni holm corrected values
# sort and correct: smallest value multiplied by n, second smallest by n-1 etc.
p_values = [p_ac, p_mi, p_co, p_acmi, p_acco, p_mico]
np.sort(p_values)

p_acco * 6
p_ac * 5
p_mi * 4
p_mico * 3
p_acmi * 2
p_co


# %% difference plots MPE adaptation
fig, ax = plt.subplots(figsize=(5, 8))

palette = ['crimson', 'darkkhaki', 'darkorange']

# Create violin plots without mini-boxplots inside.
ax = sns.violinplot(y='MPE_reduc', x='Group_x', data=df_no,
                    hue='Group_x', palette=palette, linewidth=0,
                    cut=0.7, inner=None, dodge=False, order=['C', 'M', 'A'])

# Clip the right half of each violin.
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                       transform=ax.transData))

num_items = len(ax.collections)
sns.stripplot(y='MPE_reduc', x='Group_x', hue='Group_x', palette=palette,
              edgecolor='white', linewidth=2, jitter=0.15,
              data=df_no, s=10, order=['C', 'M', 'A'], alpha=0.6)

sns.boxplot(y='MPE_reduc', x='Group_x', data=df_no, width=0.2,
            showfliers=False, showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='black',
                           markersize=12, zorder=3, markeredgecolor='white'),
            boxprops=dict(facecolor=(0, 0, 0, 0),
                          linewidth=2.5, zorder=3, edgecolor='black'),
            whiskerprops=dict(linewidth=2.5, color='black'),
            capprops=dict(linewidth=2.5, color='black'),
            medianprops=dict(linewidth=2.5, color='black'), order=['C', 'M', 'A'])

plt.axhline(0, color='Black', linewidth=1, linestyle='--')

ax.set_ylim((-4, 1.5))
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
           bbox_to_anchor=(0.05, 0.18),
           fancybox=True,
           edgecolor='white',
           framealpha=1
           )

fig.savefig(op.join(config.paths['plots'], 'MPE_reduc_base_wash.svg'))

# %% check different timing of MI group

# adaptation MPE
timing_mi = stat_adapt.loc[stat_adapt.Group == 'M', ['VP', 'MPE_reduc']]  # 3, 9

timing_mi_400 = timing_mi.loc[(timing_mi.VP == '0003') |
                              (timing_mi.VP == '0009'), :]

timing_mi_600 = timing_mi.loc[(timing_mi.VP != '0003') &
                              (timing_mi.VP != '0009'), :]


mean_active_400 = np.mean(timing_mi_400.MPE_reduc)
mean_active_600 = np.mean(timing_mi_600.MPE_reduc)
std_active_600 = np.std(timing_mi_600.MPE_reduc)

mi_timing_difference = (mean_active_400 - mean_active_600) / std_active_600

# %% washout effect
active = df_no.loc[df_no.Group_y == 'M', ['VP', 'MPE_reduc']]

active_400 = active.loc[(active.VP == '0003') |
                        (active.VP == '0009'), :]

active_600 = active.loc[(active.VP != '0003') &
                        (active.VP != '0009'), :]


mean_active_400 = np.mean(active_400.MPE_reduc)
mean_active_600 = np.mean(active_600.MPE_reduc)
std_active_600 = np.std(active_600.MPE_reduc)

active_difference = (mean_active_400 - mean_active_600) / std_active_600
