import numpy as np
import numba
from scipy.cluster.hierarchy import average, cut_tree
from scipy.spatial.distance import pdist, squareform
from scipy.special import binom


def cluster_with_mpear(X, max_clusters=None, lock_clusters=False):
    '''
    Args:
        X : (array) An array with as many rows as (post-burnin) MCMC iterations and columns as data points.
    '''
    X = np.array(X).T

    dist_mat = pdist(X, metric='hamming')

    sim_mat = 1 - squareform(dist_mat)

    Z = average(dist_mat)

    max_pear = 0

    best_cluster_labels = _get_flat_clustering(Z, 1)

    if max_clusters is None:
        max_clusters = len(X) + 1

    else:
        max_clusters = min(max_clusters, len(X))

    max_clusters = max(max_clusters, 1)

    if lock_clusters:
        cluster_labels = _get_flat_clustering(Z, max_clusters)
        return cluster_labels

    for i in range(2, max_clusters + 1):
        cluster_labels = _get_flat_clustering(Z, i)

        pear = _compute_mpear(cluster_labels, sim_mat)

        if pear > max_pear:
            max_pear = pear

            best_cluster_labels = cluster_labels

    return best_cluster_labels


def _get_flat_clustering(Z, number_of_clusters):
    N = len(Z) + 1

    if number_of_clusters == N:
        return np.arange(1, N + 1)

    return np.squeeze(cut_tree(Z, n_clusters=number_of_clusters))


def _compute_mpear(cluster_labels, sim_mat):
    N = sim_mat.shape[0]

    ind_mat = _get_indicator_matrix(cluster_labels)

    i_s = np.tril(ind_mat * sim_mat, k=-1).sum()

    i = np.tril(ind_mat, k=-1).sum()

    s = np.tril(sim_mat, k=-1).sum()

    c = binom(N, 2)

    z = (i * s) / c

    num = i_s - z

    den = 0.5 * (i + s) - z

    return num / den


@numba.jit(cache=True, nopython=True)
def _get_indicator_matrix(cluster_labels):
    N = len(cluster_labels)

    I = np.zeros((N, N))

    for i in range(N):
        for j in range(i):
            if cluster_labels[i] == cluster_labels[j]:
                I[i, j] = 1

            else:
                I[i, j] = 0

    return I + I.T
