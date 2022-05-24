import numpy as np
import numba
from numba import jit
from src.inference.beta_utils import *
from scipy.stats import uniform
import math


def update_x(t_iter, trace, patient_data, list_of_potts, beta_model, mu_priors, a0, b0,
             beta_strength, y_data, n_cache, ybar_cache, sumsqr_cache, truth_ids=None, is_anchor=False):
    """
        Updating patient labels
    """
    check_valid_model(beta_model)
    list_of_num_cells = np.array([patient_data[patient].get_size() for patient in range(len(patient_data))])
    xs = trace.get_x_curr()  # First column of x_trace.

    if truth_ids is not None:
        patient_loop = list(set(range(len(patient_data))) - set(truth_ids))
    else:
        patient_loop = list(range(len(patient_data)))

    if is_anchor:
        patient_loop.remove(max(patient_loop))

    # Loop over all patients
    for patient in patient_loop:
        # Obtain c and beta
        patient_class = int(trace.get_c_curr()[patient])
        beta = trace.get_beta_curr()[patient_class, :]

        if beta_model == TWO_K:
            # Sample x
            collapsed_gibbs_sample_xs_2k(patient=patient, list_of_num_cells=list_of_num_cells,
                                         num_cell_types=trace.cell_type, y=y_data, X=xs,
                                         neighbours=list_of_potts[patient].neighbours,
                                         beta=beta, beta_strength=beta_strength,
                                         mu_priors=mu_priors, a0=a0, b0=b0,
                                         n_cache=n_cache, ybar_cache=ybar_cache, sumsqr_cache=sumsqr_cache)

        if beta_model == ONE_PARAM:
            collapsed_gibbs_sample_xs_1p(patient=patient, list_of_num_cells=list_of_num_cells,
                                         num_cell_types=trace.cell_type, y=y_data, X=xs,
                                         neighbours=list_of_potts[patient].neighbours,
                                         beta=beta, beta_strength=beta_strength,
                                         mu_priors=mu_priors, a0=a0, b0=b0,
                                         n_cache=n_cache, ybar_cache=ybar_cache, sumsqr_cache=sumsqr_cache)

    trace.update_x_all_patients(t_iter)


@jit(nopython=True)
def collapsed_gibbs_sample_xs_2k(patient, list_of_num_cells, num_cell_types, y, X, mu_priors, a0, b0,
                                 neighbours, beta, beta_strength, n_cache, ybar_cache, sumsqr_cache):
    """
        Collapsing gibbs to avoid inference on mu and variance through MH, X is updated in-place here
    """
    # current starting positions in y
    y_start_pos = 0
    for i in range(patient):
        y_start_pos += list_of_num_cells[i]

    # total number of cells of this patient
    num_cells = list_of_num_cells[patient]

    # likelihood trace
    likelihood_trace = np.zeros((num_cells, num_cell_types))

    # update label of cell of patient
    for node in range(num_cells):
        # Populate log potentials, Q
        log_node_energies = np.zeros(num_cell_types)

        for label in range(num_cell_types):
            # Remembering current labels and modifying the labels
            current_label = X[patient, node]
            X[patient, node] = label

            # Updating the cache of n, y_bar, and sumsqr
            update_ybar_ssq_n(n_vec=n_cache, ybar_mtx=ybar_cache, sumsqr=sumsqr_cache,
                              z=current_label, k=X[patient, node], yj=y[y_start_pos + node])

            # Calculate log node energy
            log_node_energy = 0
            for node_v in neighbours[node]:
                if node_v == -1:
                    break
                i, j = int(X[patient, node]), int(X[patient, node_v])
                nonlin = lambda x, y: 1 / (1 - x * y)
                if i == j:
                    b = beta[i, 0] + beta[j, 0]
                else:
                    b = beta[i, 1] + beta[j, 1]
                log_node_energy += b * beta_strength / 2

            # Calculate log marginal likelihood
            log_marginal_likelihood = update_log_marginal_likelihood(n_vec=n_cache, ybar_mtx=ybar_cache,
                                                                     mu_priors=mu_priors,
                                                                     a0=a0, b0=b0,
                                                                     sumsqr=sumsqr_cache)
            # Sum up to update Q
            log_node_energies[label] = log_node_energy + log_marginal_likelihood

            # Likelihood trace update
            likelihood_trace[node, label] = log_marginal_likelihood

        # Manual log(sum(exp()))
        mx = np.max(log_node_energies)
        ds = log_node_energies - mx
        sum_of_exp = np.exp(ds).sum()
        log_norm_constant = mx + np.log(sum_of_exp)

        # Normalize it and draw new label
        log_prob = log_node_energies - log_norm_constant
        prob = np.exp(log_prob)

        # Reason for this is because, the previous line somehow didn't sum up to 1, hence normalize again
        prob /= np.sum(prob)

        # Update the label at node in patient
        new_label = np.arange(num_cell_types)[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
        X[patient, node] = new_label

        # Update the cache
        update_ybar_ssq_n(n_vec=n_cache, ybar_mtx=ybar_cache, sumsqr=sumsqr_cache,
                          z=(num_cell_types - 1), k=new_label, yj=y[y_start_pos + node])

    return likelihood_trace


@jit(nopython=True)
def collapsed_gibbs_sample_xs_1p(patient, list_of_num_cells, num_cell_types, y, X, mu_priors, a0, b0,
                                 neighbours, beta, beta_strength, n_cache, ybar_cache, sumsqr_cache):
    """
        Collapsing gibbs to avoid inference on mu and variance through MH, X is updated in-place here
    """
    # current starting positions in y
    y_start_pos = 0
    for i in range(patient):
        y_start_pos += list_of_num_cells[i]

    # total number of cells of this patient
    num_cells = list_of_num_cells[patient]

    # update label of cell of patient
    for node in range(num_cells):
        # Populate log potentials, Q
        log_node_energies = np.zeros(num_cell_types)

        for label in range(num_cell_types):
            # Remembering current labels and modifying the labels
            current_label = X[patient, node]
            X[patient, node] = label

            # Updating the cache of n, y_bar, and sumsqr
            update_ybar_ssq_n(n_vec=n_cache, ybar_mtx=ybar_cache, sumsqr=sumsqr_cache,
                              z=current_label, k=X[patient, node], yj=y[y_start_pos + node])

            # Calculate log node energy
            log_node_energy = 0
            for node_v in neighbours[node]:
                if node_v == -1:
                    break
                i, j = int(X[patient, node]), int(X[patient, node_v])
                if i == j:
                    b = beta[0]
                else:
                    b = 1 - beta[0]
                log_node_energy += b * beta_strength

            # Calculate log marginal likelihood
            log_marginal_likelihood = update_log_marginal_likelihood(n_vec=n_cache, ybar_mtx=ybar_cache,
                                                                     mu_priors=mu_priors,
                                                                     a0=a0, b0=b0,
                                                                     sumsqr=sumsqr_cache)
            # Sum up to update Q
            log_node_energies[label] = log_node_energy + log_marginal_likelihood

        # Manual log(sum(exp()))
        mx = np.max(log_node_energies)
        ds = log_node_energies - mx
        sum_of_exp = np.exp(ds).sum()
        log_norm_constant = mx + np.log(sum_of_exp)

        # Normalize it and draw new label
        log_prob = log_node_energies - log_norm_constant
        prob = np.exp(log_prob)

        # Reason for this is because, the previous line somehow didn't sum up to 1, hence normalize again
        prob /= np.sum(prob)

        # Update the label at node in patient
        new_label = np.arange(num_cell_types)[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
        X[patient, node] = new_label

        # Update the cache
        update_ybar_ssq_n(n_vec=n_cache, ybar_mtx=ybar_cache, sumsqr=sumsqr_cache,
                          z=(num_cell_types - 1), k=new_label, yj=y[y_start_pos + node])

    return


@jit(nopython=True)
def update_log_marginal_likelihood(n_vec, ybar_mtx, sumsqr, mu_priors, a0, b0, k0=0.1):
    """
        Calculate the log marginal likelihood
    """
    # Normal prior params    b0 = prec_mean / prec_var     a0 = prec_mean * b0
    log_2pi = np.log(2 * np.pi)

    # Calculate posterior
    kn = k0 * mu_priors + np.expand_dims(n_vec, 1)
    # kn = k0 * np.ones_like(mu_priors) + np.expand_dims(n_vec, 1)
    an = a0 + np.expand_dims(n_vec, 1) / 2

    A = (ybar_mtx - mu_priors) ** 2
    coef = k0 * n_vec
    term3 = A * coef.reshape(-1, 1) / kn / 2
    bn = b0 + 0.5 * sumsqr + term3

    z = 0.5 * log_2pi + log_gamma(an) - 0.5 * np.log(kn)
    z = np.sum(z - np.log(bn) * an, axis=1)

    # Find all indicies of n_vec that are 0 and apply the below function, replace the values in z
    indicies = np.where(n_vec == 0)[0]
    z0 = np.sum((0.5 * log_2pi + log_gamma(a0) - a0 * np.log(b0) - 0.5 * np.log(k0)), axis=1)
    z[indicies] = z0[indicies]
    z += n_vec * 0.5 * log_2pi - z0

    return np.sum(z)


@jit(nopython=True)
def update_ybar_ssq_n(n_vec, ybar_mtx, sumsqr, z, k, yj):
    """
    Parameters
    ----------
    n_vec - (mod-in-place) A vector of counts of cells in each cluster (n_vec length equals to the number of clusters)
    ybar_mtx -(mod-in-place)  A matrix of Ybars of dimension NumClusters by NumChannels
    cluster_curr - curr cluster assignment
    cluster_new - new cluster assignment
    patient - patient index
    cell - cell index (given patient)
    Y - expression matrix (stacked)
    yj - y-value of cell
    Returns
    -------
    A comment on notation:
    Assume we're updating Xj = z to Xj = k
    Then we will update
    ybark: add a term
    ybarz: remove a term
    sumsqrk: add a term
    sumsqrz: remove a term
    nk: add 1
    nz: minus 1
    prefix 'updated' denotes updated value.
    """
    # Compute new_ybar_curr
    if z == k:
        return
    ybarz = ybar_mtx[z]
    nz = n_vec[z]
    updated_nz = nz - 1
    updated_ybarz = ybarz * np.nan
    if updated_nz > 0:
        # Check for nans
        if np.isnan(yj).any():
            yj[np.isnan(yj)] = 0
        updated_ybarz = (ybarz * nz - yj) / updated_nz

    # Compute new_ybar_new
    ybark = ybar_mtx[k]
    nk = n_vec[k]
    updated_nk = nk + 1
    updated_ybark = np.ones_like(ybark) * yj
    if updated_nk > 1:
        updated_ybark = (ybark * nk + yj) / updated_nk

    # Compute new_sumsqr
    sumsqrz = sumsqr[z]
    sumsqrk = sumsqr[k]
    updated_sumsqrz = sumsqrz + nz * ybarz ** 2 - yj ** 2 - updated_nz * updated_ybarz ** 2
    updated_sumsqrk = sumsqrk + nk * ybark ** 2 + yj ** 2 - updated_nk * updated_ybark ** 2
    if updated_nk == 1:
        updated_sumsqrk = np.zeros_like(updated_sumsqrk)

    # Update/modify inputs in-place
    ybar_mtx[z] = updated_ybarz
    ybar_mtx[k] = updated_ybark

    n_vec[z] = updated_nz
    n_vec[k] = updated_nk
    sumsqr[z] = updated_sumsqrz
    sumsqr[k] = updated_sumsqrk


def update_beta(t_iter, dmh_iter, trace, prior, proposal, patient_data, beta_model, use_sumproduct=False):
    """
        Updating beta
    """
    check_valid_model(beta_model)

    # Get patient classes
    patient_classes = trace.get_c_curr()

    # Update beta for each beta class
    for patient_class in range(trace.patient_class):
        # Get patients associated with this set of beta
        patients = np.arange(trace.patient_count)[patient_classes == patient_class]

        # Get current beta with this patient class
        beta = trace.get_beta_curr()[patient_class, :]

        # Draw a new beta via DMH
        if beta_model == TWO_K:
            new_beta = dmh_sample_beta_2k(dmh_iter, trace, prior, proposal, patient_data, beta, patients)
        if beta_model == ONE_PARAM:
            new_beta = dmh_sample_beta_1p(dmh_iter, trace, prior, proposal, patient_data, beta, patients, beta_model)

        # Update trace.
        trace.update_beta(t_iter, patient_class, new_beta)

    return new_beta


def dmh_sample_beta_2k(dmh_iter, trace, prior, proposal, patient_data, beta_curr, patients):
    """
        Infer on beta using Double Metropolis Hastings for 2k model
    """
    beta = np.copy(beta_curr)

    # Looping through the clusters
    k = beta.shape[0]
    for i in range(k):
        tmp_beta = np.copy(beta[i, :])  # OG beta
        sexy_beta = proposal["beta"].propose(tmp_beta)  # The new babe TODO

        # Aggregate sufficient stats
        aux_sufficient_stat = np.zeros(k * 2)
        sufficient_stat = np.zeros(k * 2)

        # Looping through all patients
        for patient in patients:
            n_cells = patient_data[patient].get_size()
            x = trace.get_x_curr()[patient, :n_cells]
            hot_potts = prior["x"].get(patient)
            beta[i, :] = np.copy(sexy_beta)
            hot_potts.beta = np.copy(beta)  # Revert back!!

            # Sufficient statistics
            patient_aux_sufficient_stat = hot_potts.auxiliary_sufficient_statistics(x, dmh_iter)
            aux_sufficient_stat += patient_aux_sufficient_stat
            patient_sufficient_stat = hot_potts.sufficient_statistics(x)
            sufficient_stat += patient_sufficient_stat
            beta[i, :] = np.copy(tmp_beta)  # Reverted.
            hot_potts.beta = np.copy(beta)

        # Proposed factors
        beta[i, :] = np.copy(sexy_beta)  # Revert back!!
        log_prior_p = np.sum(prior["beta"].log_density(beta[:, 0]))
        proposed_beta_flat = np.concatenate((beta[:, 0], beta[:, 1]))
        log_fauxp = proposed_beta_flat.dot(aux_sufficient_stat)
        log_fxp = proposed_beta_flat.dot(sufficient_stat)
        beta[i, :] = np.copy(tmp_beta)  # Reverted.

        # Current factors
        log_prior_t = np.sum(prior["beta"].log_density(beta[:, 0]))
        beta_flat = np.concatenate((beta[:, 0], beta[:, 1]))
        log_fauxt = beta_flat.dot(aux_sufficient_stat)
        log_fxt = beta_flat.dot(sufficient_stat)

        log_ratio = log_fauxt - log_fxt + log_fxp - log_fauxp + log_prior_p - log_prior_t
        log_uniform_rv = np.log(uniform.rvs())
        reject = log_uniform_rv > log_ratio

        if not reject:
            beta[i, :] = np.copy(sexy_beta)

    return np.copy(beta)


def dmh_sample_beta_1p(dmh_iter, trace, prior, proposal, patient_data, beta, patients, beta_model):
    """
        Infer on beta using Double Metropolis Hastings for single beta model
    """
    proposed_beta = proposal["beta"].propose(beta)
    if beta_model == ONE_PARAM:
        proposed_beta = propose_beta_1p(beta, proposal, trace)

    # Aggregate sufficient stats
    aux_sufficient_stat = np.zeros(beta.shape[0])
    sufficient_stat = np.zeros(beta.shape[0])

    # Loop through all patients
    for patient in patients:
        n_cells = patient_data[patient].get_size()
        x = trace.get_x_curr()[patient, :n_cells]
        andy_potts = prior["x"].get(patient)
        andy_potts.beta = np.copy(proposed_beta)

        # Sufficient statistics
        patient_aux_sufficient_stat = andy_potts.auxiliary_sufficient_statistics(x, dmh_iter)
        aux_sufficient_stat += patient_aux_sufficient_stat
        patient_sufficient_stat = andy_potts.sufficient_statistics(x)
        sufficient_stat += patient_sufficient_stat
        andy_potts.beta = np.copy(beta)

    # Proposed factors
    log_prior_p = prior["beta"].log_density(proposed_beta)
    if beta_model == ONE_PARAM:
        log_prior_p = prior["beta"].log_density(proposed_beta[0])
    log_fauxp = proposed_beta.dot(aux_sufficient_stat)
    log_fxp = proposed_beta.dot(sufficient_stat)

    # Current factors
    log_prior_t = prior["beta"].log_density(beta)
    if beta_model == ONE_PARAM:
        log_prior_t = prior["beta"].log_density(beta[0])
    log_fauxt = beta.dot(aux_sufficient_stat)
    log_fxt = beta.dot(sufficient_stat)

    log_ratio = log_fauxt - log_fxt + log_fxp - log_fauxp + log_prior_p - log_prior_t
    log_uniform_rv = np.log(uniform.rvs())
    reject = log_uniform_rv > log_ratio

    if reject:
        return np.copy(beta)
    else:
        return np.copy(proposed_beta)


def propose_beta_1p(beta, proposal, trace):
    """
        Proposed single beta model
    """
    # Get the diagonal beta
    beta_diag = beta[:2]

    # Propose new diagonal and off diagonal
    proposed_beta_diag = proposal["beta"].propose(beta_diag)

    # Fill up a beta with off diagonals
    proposed_beta = np.full(beta.shape[0], proposed_beta_diag[1])

    #  Put into beta array that is size of beta
    diag_indices = [int(trace.cell_type * i - (i - 1) * i / 2 + i - i) for i in range(trace.cell_type)]
    for index in diag_indices:
        proposed_beta[index] = proposed_beta_diag[0]

    return proposed_beta


@numba.vectorize(["float64(float64)", "int64(float64)"])
def log_gamma(x):
    return math.lgamma(x)
