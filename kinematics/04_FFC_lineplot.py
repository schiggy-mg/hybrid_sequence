#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lineplot of FFC over time for all groups (Fig. 2E)

"""

from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.collections as collections
import os.path as op

import sys
config_path = '/Users/mg/Desktop/hybrid_movement'
sys.path.append(config_path)

import config

# %%
data_folder = op.join(config.paths['kinarm'], 'CLAMP_Block/')
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

    df_append.append(df)

df = pd.concat(df_append, axis=0)

df.reset_index(drop=True, inplace=True)

# exclude participant
df = df.loc[df.VP != '0004', :]
df = df.loc[df.VP != '0007', :]
df = df.loc[df.VP != '0036', :]
df = df.loc[df.VP != '0039', :]
df = df.loc[df.VP != '0047', :]


# %% unimanual groups

df_control = df.loc[df.Group == 'C', :]
df_active1 = df.loc[df.Group == 'A', :]
df_MI = df.loc[df.Group == 'M', :]

# %%
palette = [['darkkhaki', 'darkkhaki', 'darkkhaki'],
           ['darkorange', 'darkorange', 'darkorange'],
           ['crimson', 'crimson', 'crimson'],]

xrange1 = [(0, 6.1)]
xrange2 = [(56.9, 60)]
yrange = (-20, 120)

c1 = collections.BrokenBarHCollection(xrange1, yrange, facecolor='grey', alpha=0.2)
c2 = collections.BrokenBarHCollection(xrange2, yrange, facecolor='grey', alpha=0.2)

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x='Block_Run_Count', y='Slope', hue='Phase',
             estimator='mean', ci=95, n_boot=10000,
             palette=palette[0],
             data=df_control, legend=True)
sns.lineplot(x='Block_Run_Count', y='Slope', hue='Phase',
             estimator='mean', ci=95, n_boot=10000,
             palette=palette[1],
             data=df_MI, legend=False)
sns.lineplot(x='Block_Run_Count', y='Slope', hue='Phase',
             estimator='mean', ci=95, n_boot=10000,
             palette=palette[2],
             data=df_active1, legend=False)
plt.grid(axis='y')
plt.hlines(0, 1, 60, colors='Black', linewidth=0.5)

ax.set_xlim((1, 60))
ax.set_ylim((-20, 100))
ax.add_collection(c1)
ax.add_collection(c2)
plt.xlabel('Block')
plt.ylabel('Adaptation [%]')

handle1 = mpatches.Patch(color=palette[0][0], label='control')
handle2 = mpatches.Patch(color=palette[1][1], label='1 hand active')
handle3 = mpatches.Patch(color=palette[2][2], label='2 hand active')

plt.legend(handles=[handle1, handle2, handle3],
           title='Group',
           loc='upper left',
           facecolor='White',
           fancybox=True,
           edgecolor='white',
           framealpha=1)

fig.savefig(op.join(config.paths['plots'], 'FFC_group.svg'))

