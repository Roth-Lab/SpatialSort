import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.inference.beta_utils import *


def plot_patient_profile(run_name, x_pred, exp_csv, output_dir, vmin=None, vmax=None):
    # Read the normalized expression data
    exp_mtx = pd.read_csv(exp_csv)

    # Get patient ids in data
    file_id = list(exp_mtx.columns)[0]
    ids_wanted = pd.unique(exp_mtx[file_id]).tolist()

    # Read the user labels
    cluster_count = int(max(np.hstack(x_pred)) + 1)

    # Get a color map for showing cluster belonging in heatmap
    cmap = sns.color_palette("hls", cluster_count)
    my_colors = [cmap[i] for i in range(cluster_count)]

    for i in range(len(x_pred)):
        # Get expression matrix of patient, reindex to ensure the color mapping can coordinate
        sample = exp_mtx[exp_mtx[file_id] == ids_wanted[i]]
        sample = sample.assign(labels=x_pred[i])
        sample = sample.sort_values(by="labels")

        # Create a color mapping for cluster color
        row_colors = pd.DataFrame({'Cluster': [my_colors[i] for i in sample["labels"]]})
        heat_cmap = sns.diverging_palette(240, 10, n=9)

        # Drop unwanted columns
        sample = sample.drop(columns=[file_id, "labels"]).reset_index(drop=True)

        # Add in expression heat map
        if vmin or vmax:
            g = sns.clustermap(sample, row_colors=row_colors, row_cluster=False, col_cluster=False,
                               cmap=heat_cmap, figsize=(15, 14), vmin=vmin, vmax=vmax)
        else:
            g = sns.clustermap(sample, row_colors=row_colors, row_cluster=False, col_cluster=False,
                               cmap=heat_cmap, figsize=(15, 14))
        g.fig.suptitle("Heatmap for {}".format(ids_wanted[i]))
        plt.tight_layout()

        # Add legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=my_colors[k]) for k in range(cluster_count)]
        plt.subplots_adjust(right=0.9)
        plt.legend(handles, dict(zip(range(cluster_count), my_colors)), title='Cluster',
                   bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc="upper right", borderaxespad=0.)

        # Save figure
        plt.savefig(output_dir + run_name + f"{ids_wanted[i]}_profile.png")
        plt.clf()


def plot_cell_graph(run_name, x_pred, loc_csv, output_dir, cell_assignment_dict=None):
    # Read data to get cubes for plotting
    location_list = pd.read_csv(loc_csv)
    id_name = list(location_list.columns)[0]
    unique_ids = pd.unique(location_list[id_name])
    for i in range(len(unique_ids)):
        location_matrix = location_list[location_list[id_name] == unique_ids[i]]
        location_matrix = location_matrix.drop(columns=[id_name]).to_numpy()
        if cell_assignment_dict:
            user_labels = [cell_assignment_dict[j] for j in x_pred[i]]
        else:
            user_labels = x_pred[i]
        __plot_cell_graph(location_matrix, title="{}".format(unique_ids[i]), user_labels=user_labels)
        plt.savefig(output_dir + run_name + f"{unique_ids[i]}_graph.png", bbox_inches='tight')
        plt.clf()


def __plot_cell_graph(location_matrix, title="Neighbor Graph", user_labels=None):
    # Plot the neighbor graph
    plt.figure(figsize=(10, 8))
    plt.title(title, weight="bold")
    loc_df = pd.DataFrame(location_matrix)
    loc_df = loc_df.assign(label=user_labels)
    loc_df = loc_df.sort_values(by="label")
    sns.scatterplot(data=loc_df, x=loc_df.columns[0], y=loc_df.columns[1],
                    hue=[str(i) for i in loc_df["label"]], palette="deep", legend=True)
    plt.legend(title=r"$\bf{Cell}$" + " " + r"$\bf{Type}$", bbox_to_anchor=(1, 1), frameon=False)
    plt.box(False)
    plt.axis('off')


def plot_cluster_profile(run_name, exp_mtx, x_pred, output_dir, row_cluster=False, vmin=None, vmax=None):
    # Get patient ids in data
    file_id = list(exp_mtx.columns)[0]
    ids_wanted = pd.unique(exp_mtx[file_id]).tolist()

    # Find the amount of cells in each patient, data wrangle
    exp_mtx = exp_mtx.assign(label=x_pred)

    # Max amount of clusters
    new_k_cluster = max(x_pred) + 1

    # Get a color map for cells that belong to patient
    p_cmap = sns.color_palette("hls", len(ids_wanted))
    p_colors = [p_cmap[i] for i in range(len(ids_wanted))]
    p_to_color_mapping = dict(zip(ids_wanted, range(len(ids_wanted))))

    # For each label, plot heatmap and reference to pg labels
    for i in range(new_k_cluster):
        # Get expression of a single label
        sample = exp_mtx[exp_mtx["label"] == i]

        # If cell count is 0 for a single cluster, then don't make a plot
        if sample.shape[0] <= 1:
            continue

        # Reset the index
        sample = sample.reset_index(drop=True)

        # Get a color map for blocking rows with same pg labels in heatmap
        p_to_color_code = [p_to_color_mapping[i] for i in sample[file_id]]
        patient_row_colors = pd.DataFrame({'Patient': [p_colors[i] for i in p_to_color_code]})
        row_colors = pd.concat([patient_row_colors], axis=1)
        sample = sample.drop(columns=[file_id, "label"])

        # Plot all clusters
        heat_cmap = sns.diverging_palette(240, 10, n=9)
        if vmin or vmax:
            sns.clustermap(sample, row_colors=row_colors, row_cluster=row_cluster, col_cluster=False, cmap=heat_cmap,
                           vmin=vmin, vmax=vmax)
        else:
            sns.clustermap(sample, row_colors=row_colors, row_cluster=row_cluster, col_cluster=False, cmap=heat_cmap)
        plt.title("Cluster {} Heatmap".format(i))
        plt.savefig(output_dir + run_name + "_expression_cluster_" + str(i) + ".png")
        plt.clf()


def interaction_matrix(run_name, beta_trace, beta_model, output_dir, burnin_fraction=0.5):
    """
    :param beta_trace: trace from beta
    :param burnin_fraction: fraction of burnin
    :return: interaction matrix, also saved in output
    """
    if beta_model == TWO_K:
        # get the shape of the trace
        n_iter, n_pclasses, n_clusters, beta_dim = beta_trace.shape

        # for each patient class
        for k in range(n_pclasses):
            # get point estimate of beta same and beta diff by taking the mean of the beta trace
            burnin = round(n_iter * burnin_fraction)
            beta_same = beta_trace[burnin:, k, :, 0].mean(axis=0)
            beta_diff = 1 - beta_same

            # construct the beta interaction matrix, rounded by 2
            beta_mtx = (np.repeat(beta_diff, n_clusters).reshape(n_clusters, n_clusters)
                        + np.repeat(beta_diff, n_clusters).reshape(n_clusters, n_clusters).T) / 2
            for i in range(n_clusters):
                beta_mtx[i, i] = beta_same[i]
            beta_mtx = np.round(beta_mtx, 2)

            # plot
            plt.subplots(figsize=(20, 16))
            sns.set(font_scale=0.8)
            sns.heatmap(beta_mtx, annot=True, cmap="YlGnBu")
            plt.xlabel("Cluster")
            plt.ylabel("Cluster")
            plt.title("Class {} Interaction Matrix".format(k))
            plt.savefig(output_dir + run_name + "_interaction_mtx" + str(k) + ".png")
            plt.clf()
    else:
        raise ValueError("Invalid beta model, please choose the 2k model")
