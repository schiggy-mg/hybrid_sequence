# Motor imagery and motor adaptation project

All analysis scripts used for the main analyses and plots presented in the paper "Motor imagery enhances performance beyond the imagined action" (https://doi.org/10.1073/pnas.2423642122)

Conda environment yml files are provided for kinematic and EEG analysis, respectively.

For the kinematic analyses data is first loaded in MATLAB R2021a (= version 9.10) so Kinarm functions can be used. .mat files are saved and later loaded in Python. All kinematic and EEG plots and analyses are done in Python. Please note that some plots were refined later in inkscape (e.g., axes labels etc.).

For the cluster based permutation test of the correlation of EEG and kinematic data, code from MNE python (mne.stats.permutation_cluster_test) was used and customized where necessary.

```
move_MI
│
├── config.py                                     - paths definition
├── tools_cluster_permutation.py                  - functions for cluster test
├── tools_meeg.py                                 - eeg tools
├── tools_mne_cluster_permutation.py              - original mne functions
├── MI_task_env.yml                               - YAML file for eeg analysis
├── MOVE_2.yml                                    - YAML file for kinematic analysis
├── README.md
|
├── kinematics
│   ├── preprocessing.m
│   ├── 02_calculation_MPE_FFC.py
│   ├── 02_calculation_MPE_FFC_136.py
│   ├── 03_plot trajectories
│   ├── 04_FFC_lineplot.py
│   ├── 04_MPE_lineplot.py
│   ├── 05_FFC_ttest.py
│   ├── 05_MPE_ttest.py
│   ├── 06_calculation_reaction_dwell_time.py
│   ├── 07_comparison_reaction_dwell_time.py
│   ├── 08_combine_MPE_FFC_for_correlations.py
│   ├── 09_calculate_fusion_index.py
│   └── 10_comparison_fusion_index.py
|
├── preprocessing                         - of EEG data
│   ├── 01_bad_channels_manual.py         - filtering, channel rejection
│   ├── 02_ica_automatic.py               - ICA
│   ├── 03_ica_rejection_manual.py        - ICA component rejection
│   └── 04_epoch_rejection_manual.py      - epoch rejection
|
├── eeg
│   ├── 01_ERDS_average.py                - averaged tfr, line and topo- plots
│   ├── 02_cluster_test.py                - eeg correlation w/ behavior/questionnaire
│   └── 03_reaching_task_ERDS_average.py  - averaged tfr and line and plots of
│                                           reaching task
|
└── src_reconstruction
    ├── 01_reconstruction.py              - forward solution, inverse modeling, TFR on source space data
    ├── 02_plots.py                       - source plots alpha
    └── 03_plots_supplements.py           - source plots beta
```

Preregistration and data of the project can be found here: https://osf.io/swgd9/

author: Magdalena Gippert (gippert@cbs.mpg.de)
