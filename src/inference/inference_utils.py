import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import v_measure_score
from src.preprocessing.DataCube import DataCube


def record_v_measure(t, trace, list_of_num_cells, constant=100, gmm_init=False):
    """
        Returns the v measure trace for lag 1, lag 10, GMM initialization, and constant=100
    """
    lag1_x = np.concatenate([trace.x_trace[t - 1][i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])
    lag10_x = np.concatenate([trace.x_trace[t - 10][i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])
    curr_x = np.concatenate([trace.x_curr[i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])
    trace.v_measure_lag1[t] = v_measure_score(lag1_x, curr_x)
    trace.v_measure_lag10[t] = v_measure_score(lag10_x, curr_x)
    if gmm_init:
        gmm_x = np.concatenate([trace.x_trace[1][i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])
        trace.v_measure_gmm[t] = v_measure_score(gmm_x, curr_x)
    if t > constant:
        const_x = np.concatenate(
            [trace.x_trace[constant][i, 0:list_of_num_cells[i]] for i in range(len(list_of_num_cells))])
        trace.v_measure_const[t] = v_measure_score(const_x, curr_x)


def plot_v_measure(mcmc_trace, output_dir):
    """
        Plots the v measure trace for lag 1, lag 10, GMM initialization, and constant=100
    """
    plt.figure(figsize=(10, 10))
    plt.plot(range(2, mcmc_trace.v_measure_lag1.shape[0]), mcmc_trace.v_measure_lag1[2:], "-b", label="Lag 1")
    plt.plot(range(12, mcmc_trace.v_measure_lag10.shape[0]), mcmc_trace.v_measure_lag10[12:], "-r", label="Lag 10")
    plt.plot(range(2, mcmc_trace.v_measure_gmm.shape[0]), mcmc_trace.v_measure_gmm[2:], "-g", label="GMM")
    plt.plot(range(101, mcmc_trace.v_measure_const.shape[0]), mcmc_trace.v_measure_const[101:], "-y", label="t=100")
    plt.legend(loc="upper left")
    plt.savefig(output_dir + "v_measure_trace.png")
    plt.clf()


def gene_quantiles(y_data):
    """
        Returns 0, 25, 50, 75 quantile of each gene in y_data
    """
    gene_counts = y_data.shape[1]
    quantiles = np.zeros((gene_counts, 3+1))

    for i in range(gene_counts):
        gene_column = y_data[:, i]

        # Remove nans
        gene_column = gene_column[~np.isnan(gene_column)]

        quantiles[i, 1] = np.quantile(gene_column, 0.25)
        quantiles[i, 2] = np.quantile(gene_column, 0.5)
        quantiles[i, 3] = np.quantile(gene_column, 0.75)

    return quantiles


def select_variance(y_data, prec_mean_scale, prec_var, k0=1):
    # Channel specific, rather using a fraction of a variance, mean of the prec => 1 / (x * var(y_data))
    y_var = np.var(y_data)

    # Check for nans
    if np.isnan(y_data).any():
        y_var = np.nanvar(y_data)

    # Using yvar as an estimate for norm_std = 1 / (prec_mean * k0)
    prec_mean = 1 / (prec_mean_scale * y_var) / k0
    # For gamma, prec_mean = a0 / b0, prec_var = a0 / ( b0^2 ), so b0 = prec_mean / prec_var
    b0 = prec_mean / prec_var
    a0 = prec_mean * b0
    return a0, b0


def read_anchor_data(anchor_csv, gene_names):
    # Disaggregate data anchors
    disaggregate_data = pd.read_csv(anchor_csv)
    known_id = disaggregate_data[disaggregate_data.columns[0]][0]
    known_labels = disaggregate_data["label"].tolist()

    if len(disaggregate_data.columns[1:-1]) > len(gene_names):
        raise Exception("Number of columns in anchors exceed number of columns in primary dataset.")

    new_dis_data = disaggregate_data.iloc[:, 1:-1].to_numpy()

    if len(disaggregate_data.columns[1:-1]) < len(gene_names):
        col_to_fill = [gene_names.index(i) for i in disaggregate_data.columns[1:-1]]
        new_dis_data = np.zeros((disaggregate_data.shape[0], len(gene_names)))
        new_dis_data[:] = np.nan
        for i in range(len(col_to_fill)):
            new_dis_data[:, col_to_fill[i]] = disaggregate_data.iloc[:, 1:-1].iloc[:, i]

    # build patient cube for observed data
    known_cube = DataCube(expression_mtx=new_dis_data,
                          location_mtx=None, relation=None)
    known_cube.id = known_id
    known_cube.gene_names = gene_names
    known_cube.neighbor_graph = None

    return known_cube, known_id, known_labels
