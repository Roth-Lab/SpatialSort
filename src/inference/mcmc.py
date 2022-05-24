import numpy as np
import pandas as pd
from src.distributions.SimplexProposalDistribution import SimplexProposalDistribution
from src.distributions.BetaDistribution import BetaDistribution
from src.distributions.HotPotts import HotPotts
from src.inference.Trace import Trace
from src.preprocessing.data_processing import read_data
from src.inference.gibbs_sampling import update_x, update_beta
from src.inference.beta_utils import *
from src.inference.gmm import inference_gmm
from src.inference.inference_utils import record_v_measure, gene_quantiles, select_variance, read_anchor_data
from src.etc.utilities import Record


def inference_dmh(k_clusters, t_iter, dmh_iter, beta_model, output_dir,
                  expression_csv, location_csv, relation_csv, prior_matrix=None, anchor_csv=None,
                  prec_scale=0.1, save_trace=False,
                  x_update=True, beta_update=True, extra_cluster=False,
                  mu_priors=None, gmm_init=True, x_csv=None, beta_csv=None, print_interval=1):

    check_valid_model(beta_model)

    # Read data from all patients
    patient_data = read_data(expression_csv, location_csv, relation_csv)

    # Read anchor data
    if anchor_csv is not None:
        known_cube, known_id, known_labels = read_anchor_data(anchor_csv, patient_data[0].gene_names)
        patient_data.append(known_cube)

    m_genes = patient_data[0].get_expression().shape[1]
    beta_strength = m_genes

    # If anchor is true then we have one extra patient representing anchored data,
    # and cells will increase, and an extra id called "known"
    num_patient = len(patient_data)
    list_of_num_cells = [patient_data[i].get_size() for i in range(num_patient)]
    list_of_ids = [patient_data[i].id for i in range(num_patient)]

    if extra_cluster:
        num_clusters = k_clusters + 1
    else:
        num_clusters = k_clusters

    # Reshape y for numerical computation
    y_data = np.vstack([cube.get_expression() for cube in patient_data])

    # Initialize beta priors based on model
    beta_prior = BetaDistribution(a=1, b=1)

    # Set parameters for collapsed Gibbs
    proposed_a0, proposed_b0 = select_variance(y_data=y_data, prec_mean_scale=prec_scale, prec_var=0.05, k0=1)
    larger_a0, larger_b0 = select_variance(y_data=y_data, prec_mean_scale=0.1, prec_var=100, k0=1)
    a0 = np.ones((num_clusters, m_genes)) * larger_a0
    b0 = np.ones((num_clusters, m_genes)) * larger_b0

    # Garbage cluster
    if extra_cluster:
        a0[k_clusters] = larger_a0
        b0[k_clusters] = larger_b0

    # Initialize other priors
    quantiles = gene_quantiles(y_data)
    magnitude_dict = dict({-1: 0, 0: 1, 1: 2, 2: 3})

    if mu_priors is None:
        mu_priors = np.zeros((num_clusters, m_genes))
        if prior_matrix is None:
            prior_matrix = np.ones((num_clusters, m_genes))
            for k in range(k_clusters):
                for m in range(m_genes):
                    idx = magnitude_dict[prior_matrix[k, m]]
                    mu_priors[k, m] = quantiles[m, idx]
        else:
            # Find quantiles: col 0 is always 0, col 1 is 25%, col 2 is 50%, col 3 is 75%
            for k in range(k_clusters):
                for m in range(m_genes):
                    idx = magnitude_dict[prior_matrix[k, m]]
                    mu_priors[k, m] = quantiles[m, idx]
                    if idx != -1:
                        a0[k, m] = proposed_a0
                        b0[k, m] = proposed_b0
        # Garbage cluster
        if extra_cluster:
            for m in range(m_genes):
                mu_priors[k_clusters, m] = 0

    priors = dict({'mu': mu_priors, 'beta': beta_prior, 'x': {}})  # Each patient has a different HotPotts() prior

    # Initialize x priors, with each patient having a distinct HotPotts prior
    if anchor_csv is not None:
        patients_needing_priors = num_patient - 1
    else:
        patients_needing_priors = num_patient
    for i in range(patients_needing_priors):
        priors.get("x")[i] = HotPotts(neighbour_graph=patient_data[i].get_neighbor_graph(),
                                      number_clusters=num_clusters, beta=None, beta_model=beta_model,
                                      beta_strength=beta_strength)

    # Initialize proposals
    proposals = dict({'beta': SimplexProposalDistribution(scale=0.05)})

    # Initialize trace
    c_classes = 1
    trace = Trace(t_iter, num_clusters, m_genes, c_classes, num_patient, list_of_num_cells,
                  beta_model=beta_model, beta_init="not_random")

    # Initialize patient class labels at truth
    if anchor_csv is not None:
        class_labels = (num_patient - 1) * [0]
        class_labels.append(-1)
        trace.set_c_truth(np.array(class_labels))
    else:
        trace.set_c_truth(np.array(num_patient * [0]))

    # Initialize x trace via GMM at t=1, only run when doesn't know patient cell type
    if gmm_init:
        # Remove all columns with nans for gmm to work
        x_gmm, mu_gmm, cov_gmm = inference_gmm(k_clusters=num_clusters, data=y_data[:, ~np.isnan(y_data).any(axis=0)])
        list_of_idx = np.cumsum(np.insert(list_of_num_cells, 0, 0))

        # GMM sort by largest group
        value_idx = pd.Series(x_gmm).value_counts().index
        value_dict = dict(zip(value_idx, range(len(value_idx))))
        x_gmm = [value_dict[x_gmm[i]] for i in range(len(x_gmm))]

        for p in range(num_patient):
            trace.x_curr[p, :list_of_num_cells[p]] = x_gmm[list_of_idx[p]:list_of_idx[p + 1]]
            trace.update_x_all_patients(iter=1)

    # Initialize patient cell types at truth after first round of inference
    if x_csv is not None:
        cell_labels = pd.read_csv(x_csv)
        for i in range(num_patient):
            cell_count = patient_data[i].get_size()
            x_labels = cell_labels.to_numpy()[i, :cell_count]
            x_labels = np.where(x_labels < k_clusters-1, x_labels, k_clusters-1)
            trace.x_curr[i, :cell_count] = x_labels
            trace.x_trace[0, i, :cell_count] = x_labels

    # If beta is observed then set at truth
    if beta_csv is not None:
        trace.set_beta_truth(0, pd.read_csv(beta_csv).to_numpy())

    # If anchored data exists
    if anchor_csv is not None:
        trace.x_curr[len(list_of_ids)-1, :len(known_labels)] = known_labels
        trace.update_x_all_patients(iter=1)
        is_anchor = True
    else:
        is_anchor = False

    # Initialize cache arrays for collapsed Gibbs
    n_cache = np.zeros(num_clusters, dtype=int)
    ybar_cache = np.zeros((num_clusters, y_data.shape[1]), dtype=float)
    sumsqr_cache = np.zeros((num_clusters, y_data.shape[1]), dtype=float)

    # Reshape to flattened x trace
    x_stack = np.concatenate([trace.x_curr[i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])

    # Perform caching: for each cluster label, calculate the y_bar, sum_sqr, and total count of cells in each cluster
    for k in range(num_clusters):
        indices = np.where(x_stack == k)[0]
        y_of_k = y_data[indices]
        n_cache[k] = y_of_k.shape[0]
        if np.isnan(y_of_k).any():
            y_bar = np.nanmean(y_of_k, axis=0)
            ybar_cache[k] = y_bar
            sumsqr_cache[k] = np.nansum(y_of_k ** 2, axis=0) - n_cache[k] * y_bar ** 2
        else:
            y_bar = np.mean(y_of_k, axis=0)
            ybar_cache[k] = y_bar
            sumsqr_cache[k] = np.sum(y_of_k ** 2, axis=0) - n_cache[k] * y_bar ** 2

    # Perform inference
    start_iter = 2 if gmm_init is True else 1
    record = Record(t_iter)
    for t in range(start_iter, t_iter):
        record.estimate_completion(curr=t, interval=print_interval)
        # During first time running inference, update x
        if beta_update:
        # During each time running inference, update beta
            for b_t in range(trace.beta_iter_multiplier * t, trace.beta_iter_multiplier * (t + 1)):
                update_beta(t_iter=b_t, dmh_iter=dmh_iter, trace=trace, prior=priors, proposal=proposals,
                            patient_data=patient_data, beta_model=beta_model)

        if x_update:
            update_x(t_iter=t, trace=trace, patient_data=patient_data,
                     list_of_potts=priors.get("x"), beta_model=beta_model, beta_strength=beta_strength,
                     mu_priors=priors.get("mu"), a0=a0, b0=b0,
                     y_data=y_data, n_cache=n_cache, ybar_cache=ybar_cache, sumsqr_cache=sumsqr_cache,
                     is_anchor=is_anchor)

            # Record all v measure traces
            record_v_measure(t=t, trace=trace, list_of_num_cells=list_of_num_cells, constant=100, gmm_init=gmm_init)

    # Save the x trace
    if save_trace:
        trace_tracker = []
        for i in range(t_iter):
            temp = []
            trace_range = len(list_of_num_cells) - 1 if anchor_csv else len(list_of_num_cells)
            for j in range(trace_range):
                temp += list(trace.x_trace[i][j, :list_of_num_cells[j]])
            trace_tracker.append(temp)
        pd.DataFrame(np.array(trace_tracker).T).to_csv(output_dir + "x_trace.csv", index=False)

    # Save the x_curr
    temp = []
    trace_range = len(list_of_num_cells) - 1 if anchor_csv else len(list_of_num_cells)
    for i in range(trace_range):
        temp += list(trace.x_trace[t_iter-1][i, :list_of_num_cells[i]])
    pd.Series(np.array(temp), name="label").to_csv(output_dir + "x_last_iteration.csv", index=False)

    return trace
