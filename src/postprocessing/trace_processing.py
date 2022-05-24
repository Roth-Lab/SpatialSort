import numpy as np
from src.postprocessing.cluster import cluster_with_mpear


def label_estimates(list_of_n_cells, x_trace, max_clusters=40, lock_clusters=False):
    """
        Returns a point estimate of labels given a trace of labels
    """
    # Change x trace dimensions to number of iterations times the total number of cells across all patients
    x_stack = np.concatenate([x_trace[:, i, 0:list_of_n_cells[i]] for i in range(len(list_of_n_cells))], axis=1)

    # List of starting indices except for the last value
    list_of_idx = np.cumsum(np.insert(list_of_n_cells, 0, 0))

    # Max_cluster count is set here to avoid over clustering
    estimates = pear_estimate(x_stack, max_clusters=max_clusters, lock_clusters=lock_clusters)

    # Restoring the x trace dimensions
    unstacked_estimates = [estimates[list_of_idx[i]:(list_of_idx[i + 1])] for i in range(len(list_of_n_cells))]

    return unstacked_estimates


def pear_estimate(x_trace, max_clusters, burnin_fraction=0.5, lock_clusters=False):
    """
        Returns a point estimate of labels given a trace of labels using MPEAR
    """
    t_iter, n = x_trace.shape
    burnin = int(burnin_fraction * t_iter)

    # Calling outer function: cluster_with_mpear
    labels = cluster_with_mpear(X=x_trace[burnin:, :], max_clusters=max_clusters, lock_clusters=lock_clusters)

    return labels
