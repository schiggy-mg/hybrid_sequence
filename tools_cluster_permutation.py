#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions to perform a cluster based permutation test for correlations in 3D
(channel x frequency x time)

NME functions used in cluster based permutation test between groups are used here
(the original, unchanged mne functions are in tools_mne_cluster_permutation.py)
'''

import numpy as np
from tools_mne_cluster_permutation import _setup_adjacency, _cluster_indices_to_mask, _reshape_clusters, _cluster_mask_to_indices
from mne.stats import combine_adjacency


def prepare_for_cluster_test(tfr_csd_list, n_VP, n_channels, n_freq, n_time, ch_adjacency, behavior):
    '''Prepare all variables for cluster based correlation permutation test

    Parameters
    ----------
    tfr_csd_list :
        list of tfr objects
    n_VP :
        number of participants
    n_channels :
        number of channels
    n_freq :
        number of frequency bins
    n_time :
        number of time points
    ch_adjacency :
        channel adjacency
    behavior :
        variable which is correlated with EEG data
    '''
    eeg_data = np.zeros([n_VP, n_channels, n_freq, n_time])
    for VP in range(n_VP):
        eeg_data[VP, :, :, :] = tfr_csd_list[VP].data[:][:][:]

    # create 3D correlation matrix of observed data
    # this takes some minutes, for me like 2 min
    corr_matrix = np.zeros([n_channels, n_freq, n_time])
    t_matrix = np.zeros([n_channels, n_freq, n_time])
    # p_matrix = np.zeros([n_channels, n_freq, n_time])
    for c in range(n_channels):
        for f in range(n_freq):
            for t in range(n_time):
                ERD = eeg_data[:, c, f, t]
                r = np.corrcoef(ERD, behavior)[0, 1] # pearson correlation
                # r = stats.spearmanr(ERD, behavior)[0]  # spearman correlation
                corr_matrix[c, f, t] = r

                t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
                t_matrix[c, f, t] = t_value

                # p = (1 - stats.t.cdf(x=abs(t_value), df=n_VP - 2)) * 2
                # p_matrix[c, f, t] = p

    # get t matrix in correct format for mne functions
    y = t_matrix[..., np.newaxis]
    Z = np.transpose(y, (3, 2, 1, 0))

    # make a list
    Z = [Z]

    # check dimensions for each group in X (a list at this stage)
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in Z]
    # n_samples = X[0].shape[0]
    n_times = X[0].shape[1]

    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError('All samples must have the same size')

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    n_tests = X[0].shape[1]

    # t_obs is the output of the stat fun function
    # delete first dimension because samples is not needed here, t_obs needs to be 1 dimensional
    t_obs = X[0]
    t_obs = t_obs[0, :]

    if t_obs.size != np.prod(sample_shape):
        raise ValueError('t_obs.shape %s provided by stat_fun %s is not '
                        'compatible with the sample shape %s'
                        % (t_obs.shape, 3, sample_shape))


    # define adjacency matrix in 3D
    adjacency_3D = combine_adjacency(
    n_time,  # regular lattice adjacency for times
    n_freq,
    ch_adjacency,)  # custom matrix, defined before

    # setup adjacency
    adjacency = _setup_adjacency(adjacency_3D, n_tests, n_times)

    return(t_obs, n_tests, adjacency, sample_shape, eeg_data, corr_matrix )


def reformat_clusters(threshold, t_obs, sample_shape, clusters, cluster_stats, adjacency, n_tests):
    '''reformats variables for cluster based permutation test

    Parameters
    ----------
    threshold :
        initial threshold for cluster based permutation test
    t_obs :
        observed t-values
    sample_shape :
        sample shape
    clusters :
        observed clusters
    cluster_stats :
        cluster statistics
    adjacency :
        channel adjacency
    n_tests :
        number of permutations
    '''
    if isinstance(threshold, dict):
        # The stat should have the same shape as the samples
        t_obs.shape = sample_shape

        # For TFCE, return the "adjusted" statistic instead of raw scores
        t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)

        if adjacency is None or adjacency is False:
        # get indices (for tfce), so output format of clusters returns a list of tuple of ndarray,
        # where each ndarray contains the indices of locations that together form the
        # given cluster along the given dimension
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    else:
        if adjacency is not None and adjacency is not False:
        # mask: clusters returns a list of boolean arrays, each with the same shape
        # as the input data (or slices if the shape is 1D and adjacency is None),
        # with True values indicating locations that are part of a cluster.
            clusters = _cluster_indices_to_mask(clusters, n_tests)

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)

    # clusters : list of clusters, each entry is shape timepoints, freq, channels, mask with true & false
    # cluster stats: np.nd array with length of how many clusters there are, each entry is T value of cluster
    # in tfce: TFCE values of each voxel

    observed_cluster_T = abs(cluster_stats[:]).max()  # find biggest observed cluster
    observed_cluster_idx = abs(cluster_stats[:]).argmax()

    return(clusters, observed_cluster_T, observed_cluster_idx)


def permutation_test(behavior, n_channels, n_freq, n_time, n_VP, eeg_data):
    '''perform permutation test

    Behavior variable is randomly shuffled and correlation is performed.

    Parameters
    ----------
    behavior :
        variable which is correlated with EEG data
    n_channels :
        number of channels
    n_freq :
        number of frequency bins
    n_time :
        number of time points
    n_VP :
        number of participants
    eeg_data :
        eeg data
    '''
    np.random.shuffle(behavior)  # shuffle behavior values randomly
    t_matrix = np.zeros([n_channels, n_freq, n_time])

    for c in range(n_channels):
        for f in range(n_freq):
            for t in range(n_time):
                ERD = eeg_data[:, c, f, t]
                r = np.corrcoef(ERD, behavior)[0, 1] # pearson correlation
                # r = stats.spearmanr(ERD, behavior)[0]  # spearman correlation

                t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
                t_matrix[c, f, t] = t_value

    y = t_matrix[..., np.newaxis]
    Z = np.transpose(y, (3, 2, 1, 0))

    # make a list
    Z = [Z]

    # reshape
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in Z]

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]

    t_perm = X[0]
    t_perm = t_perm[0, :]

    return(t_perm)