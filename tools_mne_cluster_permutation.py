#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
functions copied from mne to do clustering based on 3D data

functions are from "mne.stats.permutation_cluster_test"

Only these subfunctions are needed because the goal is not to compare groups so
a different kind of permutation is needed
'''

import numpy as np
from scipy import sparse


def _setup_adjacency(adjacency, n_tests, n_times):
    if not sparse.issparse(adjacency):
        raise ValueError("If adjacency matrix is given, it must be a "
                         "SciPy sparse matrix.")
    if adjacency.shape[0] == n_tests:  # use global algorithm
        adjacency = adjacency.tocoo()
    else:  # use temporal adjacency algorithm
        got_times, mod = divmod(n_tests, adjacency.shape[0])
        if got_times != n_times or mod != 0:
            raise ValueError(
                'adjacency (len %d) must be of the correct size, i.e. be '
                'equal to or evenly divide the number of tests (%d).\n\n'
                'If adjacency was computed for a source space, try using '
                'the fwd["src"] or inv["src"] as some original source space '
                'vertices can be excluded during forward computation'
                % (adjacency.shape[0], n_tests))
        # we claim to only use upper triangular part... not true here
        adjacency = (adjacency + adjacency.transpose()).tocsr()
        adjacency = [
            adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
            for i in range(len(adjacency.indptr) - 1)]
    return adjacency


def _where_first(x):
        for ii in range(len(x)):
            if x[ii]:
                return ii
        return -1


def _get_buddies(r, s, neighbors, indices=None):
        buddies = list()
        # At some point we might be able to use the sorted-ness of s or
        # neighbors to further speed this up
        if indices is None:
            n_check = len(r)
        else:
            n_check = len(indices)
        for ii in range(n_check):
            if indices is None:
                this_idx = ii
            else:
                this_idx = indices[ii]
            if r[this_idx]:
                this_s = s[this_idx]
                for ni in range(len(neighbors)):
                    if this_s == neighbors[ni]:
                        buddies.append(this_idx)
                        r[this_idx] = False
                        break
        return buddies


def _get_clusters_spatial(s, neighbors):
    """Form spatial clusters using neighbor lists.
    This is equivalent to _get_components with n_times = 1, with a properly
    reconfigured adjacency matrix (formed as "neighbors" list)
    """
    # s is a vector of spatial indices that are significant, like:
    #     s = np.where(x_in)[0]
    # for x_in representing a single time-instant
    r = np.ones(s.shape, bool)
    clusters = list()
    next_ind = 0 if s.size > 0 else -1
    while next_ind >= 0:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = 0
        icount = 1  # count of nodes in the current cluster
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            # look across other vertices
            buddies = _get_buddies(r, s, neighbors[s[ind]])
            t_inds.extend(buddies)
            icount += 1
        next_ind = _where_first(r)
        clusters.append(s[t_inds])
    return clusters


def _reassign(check, clusters, base, num):
    """Reassign cluster numbers."""
    # reconfigure check matrix
    check[check == num] = base
    # concatenate new values into clusters array
    clusters[base - 1] = np.concatenate((clusters[base - 1],
                                         clusters[num - 1]))
    clusters[num - 1] = np.array([], dtype=int)


def _get_clusters_st_1step(keepers, neighbors):
    """Directly calculate clusters.
    This uses knowledge that time points are
    only adjacent to immediate neighbors for data organized as time x space.
    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.
    This algorithm creates clusters for each time point using a method more
    efficient than the standard graph method (but otherwise equivalent), then
    combines these clusters across time points in a reasonable way.
    """
    n_src = len(neighbors)
    n_times = len(keepers)
    # start cluster numbering at 1 for diffing convenience
    enum_offset = 1
    check = np.zeros((n_times, n_src), dtype=int)
    clusters = list()
    for ii, k in enumerate(keepers):
        c = _get_clusters_spatial(k, neighbors)
        for ci, cl in enumerate(c):
            check[ii, cl] = ci + enum_offset
        enum_offset += len(c)
        # give them the correct offsets
        c = [cl + ii * n_src for cl in c]
        clusters += c

    # now that each cluster has been assigned a unique number, combine them
    # by going through each time point
    for check1, check2, k in zip(check[:-1], check[1:], keepers[:-1]):
        # go through each one that needs reassignment
        inds = k[check2[k] - check1[k] > 0]
        check1_d = check1[inds]
        n = check2[inds]
        nexts = np.unique(n)
        for num in nexts:
            prevs = check1_d[n == num]
            base = np.min(prevs)
            for pr in np.unique(prevs[prevs != base]):
                _reassign(check1, clusters, base, pr)
            # reassign values
            _reassign(check2, clusters, base, num)
    # clean up clusters
    clusters = [cl for cl in clusters if len(cl) > 0]
    return clusters


def _get_clusters_st(x_in, neighbors, max_step=1):
    """Choose the most efficient version."""
    n_src = len(neighbors)
    n_times = x_in.size // n_src
    cl_goods = np.where(x_in)[0]
    if len(cl_goods) > 0:
        keepers = [np.array([], dtype=int)] * n_times
        row, col = np.unravel_index(cl_goods, (n_times, n_src))
        lims = [0]
        if isinstance(row, int):
            row = [row]
            col = [col]
        else:
            order = np.argsort(row)
            row = row[order]
            col = col[order]
            lims += (np.where(np.diff(row) > 0)[0] + 1).tolist()
            lims.append(len(row))

        for start, end in zip(lims[:-1], lims[1:]):
            keepers[row[start]] = np.sort(col[start:end])
        if max_step == 1:
            return _get_clusters_st_1step(keepers, neighbors)
        else:
            return _get_clusters_st_multistep(keepers, neighbors,
                                              max_step)
    else:
        return []


def _masked_sum(x, c):
    return np.sum(x[c])


def _get_components(x_in, adjacency, return_list=True):
    """Get connected components from a mask and a adjacency matrix."""
    from scipy import sparse
    if adjacency is False:
        components = np.arange(len(x_in))
    else:
        from scipy.sparse.csgraph import connected_components
        mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        data = adjacency.data[mask]
        row = adjacency.row[mask]
        col = adjacency.col[mask]
        shape = adjacency.shape
        idx = np.where(x_in)[0]
        row = np.concatenate((row, idx))
        col = np.concatenate((col, idx))
        data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
        adjacency = sparse.coo_matrix((data, (row, col)), shape=shape)
        _, components = connected_components(adjacency)
    if return_list:
        start = np.min(components)
        stop = np.max(components)
        comp_list = [list() for i in range(start, stop + 1, 1)]
        mask = np.zeros(len(comp_list), dtype=bool)
        for ii, comp in enumerate(components):
            comp_list[comp].append(ii)
            mask[comp] += x_in[ii]
        clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
        return clusters
    else:
        return components


def _find_clusters_1dir(x, x_in, adjacency, max_step, t_power, ndimage):
    """Actually call the clustering algorithm."""
    from scipy import sparse
    if adjacency is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            # slices
            clusters = ndimage.find_objects(labels, n_labels)
            # equivalent to if len(clusters) == 0 but faster
            if not clusters:
                sums = list()
            else:
                index = list(range(1, n_labels + 1))
                if t_power == 1:
                    sums = ndimage.sum(x, labels, index=index)
                else:
                    sums = ndimage.sum(np.sign(x) * np.abs(x) ** t_power,
                                       labels, index=index)
        else:
            # boolean masks (raveled)
            clusters = list()
            sums = np.empty(n_labels)
            for label in range(n_labels):
                c = labels == label + 1
                clusters.append(c.ravel())
                if t_power == 1:
                    sums[label] = np.sum(x[c])
                else:
                    sums[label] = np.sum(np.sign(x[c]) *
                                         np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a adjacency "
                            "to define clusters.")
        if isinstance(adjacency, sparse.spmatrix) or adjacency is False:
            clusters = _get_components(x_in, adjacency)
        elif isinstance(adjacency, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, adjacency, max_step)
        else:
            raise ValueError('adjacency must be a sparse matrix or list')
        if t_power == 1:
            sums = [_masked_sum(x, c) for c in clusters]
        else:
            sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)


def _find_clusters_1dir_parts(x, x_in, adjacency, max_step, partitions,
                              t_power, ndimage):
    """Deal with partitions, and pass the work to _find_clusters_1dir."""
    if partitions is None:
        clusters, sums = _find_clusters_1dir(x, x_in, adjacency, max_step,
                                             t_power, ndimage)
    else:
        # cluster each partition separately
        clusters = list()
        sums = list()
        for p in range(np.max(partitions) + 1):
            x_i = np.logical_and(x_in, partitions == p)
            out = _find_clusters_1dir(x, x_i, adjacency, max_step, t_power,
                                      ndimage)
            clusters += out[0]
            sums.append(out[1])
        sums = np.concatenate(sums)
    return clusters, sums


def _find_clusters(x, threshold, tail=0, adjacency=None, max_step=1,
                   include=None, partitions=None, t_power=1, show_info=False):
    """Find all clusters which are above/below a certain threshold.
    When doing a two-tailed test (tail == 0), only points with the same
    sign will be clustered together.
    Parameters
    ----------
    x : 1D array
        Data
    threshold : float | dict
        Where to threshold the statistic. Should be negative for tail == -1,
        and positive for tail == 0 or 1. Can also be an dict for
        threshold-free cluster enhancement.
    tail : -1 | 0 | 1
        Type of comparison
    adjacency : scipy.sparse.coo_matrix, None, or list
        Defines adjacency between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If adjacency is a list, it is assumed that each entry stores the
        indices of the spatial neighbors in a spatio-temporal dataset x.
        Default is None, i.e, a regular lattice adjacency.
        False means no adjacency.
    max_step : int
        If adjacency is a list, this defines the maximal number of steps
        between vertices along the second dimension (typically time) to be
        considered adjacent.
    include : 1D bool array or None
        Mask to apply to the data of points to cluster. If None, all points
        are used.
    partitions : array of int or None
        An array (same size as X) of integers indicating which points belong
        to each partition.
    t_power : float
        Power to raise the statistical values (usually t-values) by before
        summing (sign will be retained). Note that t_power == 0 will give a
        count of nodes in each cluster, t_power == 1 will weight each node by
        its statistical score.
    show_info : bool
        If True, display information about thresholds used (for TFCE). Should
        only be done for the standard permutation.
    Returns
    -------
    clusters : list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays.
    sums : array
        Sum of x values in clusters.
    """
    from scipy import ndimage
    # _check_option('tail', tail, [-1, 0, 1])

    x = np.asanyarray(x)

    if not np.isscalar(threshold):
        if not isinstance(threshold, dict):
            raise TypeError('threshold must be a number, or a dict for '
                            'threshold-free cluster enhancement')
        if not all(key in threshold for key in ['start', 'step']):
            raise KeyError('threshold, if dict, must have at least '
                           '"start" and "step"')
        tfce = True
        use_x = x[np.isfinite(x)]
        if use_x.size == 0:
            raise RuntimeError(
                'No finite values found in the observed statistic values')
        if tail == -1:
            if threshold['start'] > 0:
                raise ValueError('threshold["start"] must be <= 0 for '
                                 'tail == -1')
            if threshold['step'] >= 0:
                raise ValueError('threshold["step"] must be < 0 for '
                                 'tail == -1')
            stop = np.min(use_x)
        elif tail == 1:
            stop = np.max(use_x)
        else:  # tail == 0
            stop = max(np.max(use_x), -np.min(use_x))
        del use_x
        thresholds = np.arange(threshold['start'], stop,
                               threshold['step'], float)
        h_power = threshold.get('h_power', 2)
        e_power = threshold.get('e_power', 0.5)
        # h_power = threshold.get('h_power', 2)
        # e_power = threshold.get('e_power', 0.8)
        #print('h=2, w=0.5, STANDARD')
        if show_info is True:
            if len(thresholds) == 0:
                warn('threshold["start"] (%s) is more extreme than data '
                     'statistics with most extreme value %s'
                     % (threshold['start'], stop))
            else:
                logger.info('Using %d thresholds from %0.2f to %0.2f for TFCE '
                            'computation (h_power=%0.2f, e_power=%0.2f)'
                            % (len(thresholds), thresholds[0], thresholds[-1],
                               h_power, e_power))
        scores = np.zeros(x.size)
    else:
        thresholds = [threshold]
        tfce = False

    # include all points by default
    if include is None:
        include = np.ones(x.shape, dtype=bool)

    if tail in [0, 1] and not np.all(np.diff(thresholds) > 0):
        raise ValueError('Thresholds must be monotonically increasing')
    if tail == -1 and not np.all(np.diff(thresholds) < 0):
        raise ValueError('Thresholds must be monotonically decreasing')

    # set these here just in case thresholds == []
    clusters = list()
    sums = list()
    for ti, thresh in enumerate(thresholds):
        # these need to be reset on each run
        clusters = list()
        if tail == 0:
            x_ins = [np.logical_and(x > thresh, include),
                     np.logical_and(x < -thresh, include)]
        elif tail == -1:
            x_ins = [np.logical_and(x < thresh, include)]
        else:  # tail == 1
            x_ins = [np.logical_and(x > thresh, include)]
        # loop over tails
        for x_in in x_ins:
            if np.any(x_in):
                out = _find_clusters_1dir_parts(x, x_in, adjacency,
                                                max_step, partitions, t_power,
                                                ndimage)
                clusters += out[0]
                sums.append(out[1])
        if tfce:
            # the score of each point is the sum of the h^H * e^E for each
            # supporting section "rectangle" h x e.
            if ti == 0:
                h = abs(thresh)
            else:
                h = abs(thresh - thresholds[ti - 1])
            h = h ** h_power
            for c in clusters:
                # triage based on cluster storage type
                if isinstance(c, slice):
                    len_c = c.stop - c.start
                elif isinstance(c, tuple):
                    len_c = len(c)
                elif c.dtype == bool:
                    len_c = np.sum(c)
                else:
                    len_c = len(c)
                scores[c] += h * (len_c ** e_power)
    # turn sums into array
    sums = np.concatenate(sums) if sums else np.array([])
    if tfce:
        # each point gets treated independently
        clusters = np.arange(x.size)
        if adjacency is None or adjacency is False:
            if x.ndim == 1:
                # slices
                clusters = [slice(c, c + 1) for c in clusters]
            else:
                # boolean masks (raveled)
                clusters = [(clusters == ii).ravel()
                            for ii in range(len(clusters))]
        else:
            clusters = [np.array([c]) for c in clusters]
        sums = scores
    return clusters, sums


def _cluster_indices_to_mask(components, n_tot):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        components[ci] = np.zeros((n_tot), dtype=bool)
        components[ci][c] = True
    return components


def _cluster_mask_to_indices(components, shape):
    """Convert to the old format of clusters, which were bool arrays."""
    for ci, c in enumerate(components):
        if isinstance(c, np.ndarray):  # mask
            components[ci] = np.where(c.reshape(shape))
        elif isinstance(c, slice):
            components[ci] = np.arange(c.start, c.stop)
        else:
            assert isinstance(c, tuple), type(c)
            c = list(c)  # tuple->list
            for ii, cc in enumerate(c):
                if isinstance(cc, slice):
                    c[ii] = np.arange(cc.start, cc.stop)
                else:
                    c[ii] = np.where(cc)[0]
            components[ci] = tuple(c)
    return components

def _reshape_clusters(clusters, sample_shape):
    """Reshape cluster masks or indices to be of the correct shape."""
    # format of the bool mask and indices are ndarrays
    if len(clusters) > 0 and isinstance(clusters[0], np.ndarray):
        if clusters[0].dtype == bool:  # format of mask
            clusters = [c.reshape(sample_shape) for c in clusters]
        else:  # format of indices
            clusters = [np.unravel_index(c, sample_shape) for c in clusters]
    return clusters

