import numpy as np
import pandas as pd
from src.inference.beta_utils import *
import os


class Trace:

    def __init__(self, t_iter, k_clusters, m_genes, c_pclass, patient_dim, list_of_num_cells,
                 beta_model=None, beta_init=None, beta_iter_multiplier=5):
        # Parameters
        self.iterations = t_iter
        self.cell_type = k_clusters
        self.gene_types = m_genes
        self.patient_class = c_pclass
        self.patient_count = patient_dim
        self.list_of_n_cells = list_of_num_cells
        self.max_cell_count = max(list_of_num_cells)

        # Beta specific parameters
        check_valid_model(beta_model)
        self.beta_model = beta_model
        self.beta_iter_multiplier = beta_iter_multiplier
        if beta_model == TWO_K:
            self.beta_dim = k_clusters
        elif beta_model == ONE_PARAM:
            self.beta_dim = int(self.cell_type * (self.cell_type + 1) / 2)
        else:
            raise ValueError("Invalid beta model, please choose one of: {}".format(BETA_MODELS))

        # Trace matrices
        self.x_trace = (np.arange(t_iter * self.patient_count * self.max_cell_count) % k_clusters).reshape(
            (t_iter, self.patient_count, self.max_cell_count))
        self.c_trace = np.zeros((t_iter, self.patient_count))

        # Beta specific trace matrix
        if beta_model == TWO_K:
            self.beta_trace = np.ones((t_iter * self.beta_iter_multiplier, self.patient_class, self.beta_dim, 2))
            self.beta_trace[:, :, :, 0] *= 0.7
            self.beta_trace[:, :, :, 1] *= 0.3
            if beta_init == "random":
                for i in range(self.patient_class):
                    self.beta_trace[0, i] = np.random.rand(self.beta_dim, 2)
                    self.beta_trace[0, i] = (self.beta_trace[0, i].T / np.sum(self.beta_trace[0, i], axis=1)).T
        elif beta_model == ONE_PARAM:
            self.beta_trace = np.ones((t_iter * self.beta_iter_multiplier, self.patient_class, self.beta_dim)) / 2
            if beta_init == "random":
                self.beta_trace = np.random.rand(t_iter * self.beta_iter_multiplier, self.patient_class, self.beta_dim)
        else:
            raise ValueError("Invalid beta model, please choose one of: {}".format(BETA_MODELS))

        # Initializing current values
        self.x_curr = self.x_trace[0, :, :]
        self.c_curr = self.c_trace[0, :]

        if beta_model == TWO_K:
            self.beta_curr = self.beta_trace[0, :, :, :]
        else:  # ONE_PARAM, POTTS
            self.beta_curr = self.beta_trace[0, :, :]

        # V measure traces
        self.v_measure_lag1 = np.zeros(t_iter)
        self.v_measure_lag10 = np.zeros(t_iter)
        self.v_measure_gmm = np.zeros(t_iter)
        self.v_measure_const = np.zeros(t_iter)

    def update_beta(self, iter, p_class, value):
        if self.beta_model == TWO_K:
            self.beta_curr[p_class, :, :] = value
            self.beta_trace[iter, p_class, :, :] = value
        else:
            self.beta_curr[p_class, :] = value
            self.beta_trace[iter, p_class, :] = value

    def update_x(self, iter, patient, n_cells, value):
        self.x_curr[patient, 0:n_cells] = value
        self.x_trace[iter, patient, 0:n_cells] = value

    def update_x_all_patients(self, iter):
        self.x_trace[iter, :, :] = np.copy(self.x_curr[:, :])

    def update_c(self, iter, patient, value):
        self.c_curr[patient] = value
        self.c_trace[iter, patient] = value

    def get_beta_curr(self):
        return self.beta_curr

    def get_x_curr(self):
        return self.x_curr

    def get_c_curr(self):
        return self.c_curr

    def set_beta_truth(self, index, value):
        if self.beta_model == TWO_K:
            self.beta_curr[index] = np.vstack((value[index][:self.beta_dim], value[index][self.beta_dim:])).T
        else:
            self.beta_curr[index] = value

    def set_x_truth(self, value):
        self.x_curr = value

    def set_c_truth(self, value):
        self.c_curr = value
        self.c_trace[0, :] = value
