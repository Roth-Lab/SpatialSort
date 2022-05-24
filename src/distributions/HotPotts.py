import numpy as np
import networkx as nx
from numba import jit
from src.inference.beta_utils import *


class HotPotts:

    def __init__(self, neighbour_graph, number_clusters, beta, beta_model, beta_strength):
        check_valid_model(beta_model)

        self.neighbour_graph = neighbour_graph
        self.num_clusters = number_clusters
        self.beta = beta
        self.beta_model = beta_model
        self.beta_strength = beta_strength

        if beta_model == TWO_K:
            # This will transform in (clusters, 2) where each cluster has a same or different interaction affinity
            self.dim_betas = number_clusters
        elif beta_model == ONE_PARAM:
            # This is a flat array of beta values corresponding to a triu of an interaction matrix
            self.dim_betas = int(number_clusters * (number_clusters + 1) / 2)
        else:  # POTTS
            self.dim_betas = 1

        self.n = neighbour_graph.number_of_nodes()
        self.edges = np.array(neighbour_graph.edges(), dtype=np.int64)

        # Storing all the neighbours for each node
        max_degree = np.max([deg for (node, deg) in neighbour_graph.degree()])
        self.neighbours = np.zeros((self.n, max_degree), dtype=np.int64) - 1
        for node in range(self.n):
            nbrs = list(neighbour_graph.neighbors(node))
            for i in range(len(nbrs)):
                self.neighbours[node, i] = nbrs[i]

        # Return an oriented tree constructed from of a breadth-first-search starting at 0
        self.tree = nx.bfs_tree(neighbour_graph, 0)

    def auxiliary_sufficient_statistics(self, starting_state, iterations):
        """
        Used for parallel computation.
        """
        y = self.gibbs_sample(iterations, starting_state)
        sufficient_statistic = self.sufficient_statistics(y)
        return sufficient_statistic

    def sufficient_statistics(self, x):
        """
        Compute sufficient statistics of a given realization.
        """
        if self.beta_model == TWO_K:
            return sufficient_statistics_helper_2k(x, self.edges, self.dim_betas)
        elif self.beta_model == ONE_PARAM:
            return sufficient_statistics_helper_1p(x, self.edges, self.num_clusters, self.dim_betas)
        else:
            raise ValueError("Invalid beta model, please choose one of: {}".format(BETA_MODELS))

    def gibbs_sample(self, iterations, start_state, fix_seed=False, seed=1):
        """
        Draw a realization from the model given an initial state.
        """
        if self.beta_model == TWO_K:
            return gibbs_sample_helper_2k(iterations, self.n, self.num_clusters, start_state, self.neighbours,
                                          self.beta, fix_seed, seed, self.beta_strength)
        elif self.beta_model == ONE_PARAM:
            return gibbs_sample_helper_1p(iterations, self.n, self.num_clusters, start_state, self.neighbours,
                                          self.beta, fix_seed, seed, self.beta_strength)
        else:
            raise ValueError("Invalid beta model, please choose one of: {}".format(BETA_MODELS))


@jit(nopython=True)
def sufficient_statistics_helper_2k(x, edges, dim_betas):
    """
    Optimized helper function for computing sufficient statistics for 2k.
    """
    statistics = np.zeros(dim_betas * 2)

    # Calculate total amount of edges that are same-same or diff-diff interactions
    for u, v in edges:
        i, j = (int(x[u]), int(x[v]))
        if i == j:  # If labels are the same
            statistics[i] += 1
            statistics[j] += 1
        else:  # If labels are different
            statistics[dim_betas + i] += 1
            statistics[dim_betas + j] += 1

    return statistics / 2


@jit(nopython=True)
def sufficient_statistics_helper_1p(x, edges, k_clusters, dim_betas):
    """
    Optimized helper function for computing sufficient statistics for k squared.
    """
    statistics = np.zeros(dim_betas)

    # Calculate the number of edges in each type of interaction
    for u, v in edges:
        i, j = (x[u], x[v])
        if i > j:  # Assumption violated, swap to satisfy assumption
            i, j = (x[v], x[u])
        beta_index = int(k_clusters * i - (i - 1) * i / 2 + j - i)
        statistics[beta_index] += 1

    return statistics


@jit(nopython=True)
def gibbs_sample_helper_2k(iterations, n, k_clusters, start_state, neighbours, betas, fix_seed, seed, beta_strength):
    """
    Optimized helper function for sampling via Gibbs moves for 2k.
    """
    if fix_seed:
        np.random.seed(seed)

    current_values = np.copy(start_state)
    for iteration in range(iterations):
        for node in range(n):
            log_node_energies = np.zeros(k_clusters)
            # Populate log potentials
            for label in range(k_clusters):
                current_values[node] = label
                node_u = node
                log_node_energy = 0
                neighbours_of_node = neighbours[node]
                for node_v in neighbours_of_node:
                    if node_v == -1:
                        break
                    u_label, v_label = int(current_values[node_u]), int(current_values[node_v])

                    # 2k specific code
                    if u_label == v_label:
                        b = betas[u_label, 0] + betas[v_label, 0]
                    else:
                        b = betas[u_label, 1] + betas[v_label, 1]
                    log_node_energy += b * beta_strength / 2.0

                log_node_energies[label] = log_node_energy

            max_log_node_energy = np.max(log_node_energies)
            ds = log_node_energies - max_log_node_energy
            sum_of_exp = np.exp(ds).sum()
            log_norm_constant = max_log_node_energy + np.log(sum_of_exp)

            log_prob = log_node_energies - log_norm_constant
            prob = np.exp(log_prob)
            prob /= np.sum(prob)
            current_values[node] = np.arange(k_clusters)[
                np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
    return current_values


@jit(nopython=True)
def gibbs_sample_helper_1p(iterations, n, k_clusters, start_state, neighbours, betas, fix_seed, seed, beta_strength):
    """
    Optimized helper function for sampling via Gibbs moves.
    """
    if fix_seed:
        np.random.seed(seed)

    current_values = np.copy(start_state)
    for iteration in range(iterations):
        for node in range(n):
            log_node_energies = np.zeros(k_clusters)
            # Populate log potentials
            for label in range(k_clusters):
                current_values[node] = label
                node_u = node
                log_node_energy = 0
                neighbours_of_node = neighbours[node]
                for node_v in neighbours_of_node:
                    if node_v == -1:
                        break
                    u_label, v_label = int(current_values[node_u]), int(current_values[node_v])

                    # 1p specific code
                    if u_label == v_label:
                        b = betas[0]
                    else:
                        b = 1 - betas[0]
                    log_node_energy += b * beta_strength

                log_node_energies[label] = log_node_energy

            max_log_node_energy = np.max(log_node_energies)
            ds = log_node_energies - max_log_node_energy
            sum_of_exp = np.exp(ds).sum()
            log_norm_constant = max_log_node_energy + np.log(sum_of_exp)

            log_prob = log_node_energies - log_norm_constant
            prob = np.exp(log_prob)
            prob /= np.sum(prob)
            current_values[node] = np.arange(k_clusters)[
                np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
    return current_values
