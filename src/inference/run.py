"""
Main function for SpatialSort

@author: Eric Lee 2021
"""
import os
import numpy as np
import pandas as pd
from src.etc.utilities import date_time
from src.inference.mcmc import inference_dmh
from src.postprocessing.trace_processing import label_estimates
from src.inference.beta_utils import *
from src.inference.inference_utils import plot_v_measure


def run(exp_csv,
        loc_csv,
        rel_csv,
        num_clusters,
        out_dir,
        num_iters=500,
        seed=None,
        prior_csv=None,
        anchor_csv=None,
        prec_scale=0.1,
        save_trace=False
        ):

    # If there's a seed use the seed
    if seed is not None:
        np.random.seed(seed)

    # If there is a prior matrix, use it
    if prior_csv:
        prior_matrix = pd.read_csv(prior_csv).to_numpy()
    else:
        prior_matrix = None

    # Create file directory
    output_dir = out_dir + "/" + "k" + str(num_clusters) \
                 + "_p" + str(1 if prior_csv else 0) \
                 + "_a" + str(1 if anchor_csv else 0) \
                 + "_s" + str(seed) \
                 + "_" + date_time()
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    print(f"Saving at {output_dir}")
    mcmc_trace = inference_dmh(
        k_clusters=num_clusters,
        t_iter=num_iters,
        dmh_iter=5,
        beta_model=TWO_K,
        output_dir=output_dir,
        expression_csv=exp_csv,
        location_csv=loc_csv,
        relation_csv=rel_csv,
        prior_matrix=prior_matrix,
        anchor_csv=anchor_csv,
        prec_scale=prec_scale,
        save_trace=save_trace
    )

    # Plot v_measure trace
    plot_v_measure(mcmc_trace, output_dir)

    # Estimate the labels
    xhat = label_estimates(mcmc_trace.list_of_n_cells, mcmc_trace.x_trace,
                           max_clusters=num_clusters,
                           lock_clusters=True if anchor_csv else False)

    # Save predicted x labels
    xhat_flat = [x for row in xhat for x in row]
    pd.Series(xhat_flat, name="label").to_csv(output_dir + "x_hat.csv", index=False)

    return xhat_flat
