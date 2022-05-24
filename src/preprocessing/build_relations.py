import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform, pdist


def build_relations(location_csv, output_csv, threshold=30):
    """
    :param location_csv: the name of your input file, should have three columns: one named "file_id", and the other two are X and Y
    :param output_csv: the name of your output file
    :param threshold: a threshold to determine how close you consider cells interacting
    :return: output a csv file of relations
    """
    location_matrix = pd.read_csv(location_csv)
    id_name = list(location_matrix.columns)[0]
    unique_ids = pd.unique(location_matrix[id_name])
    relation = []
    for id in unique_ids:
        subset_mtx = location_matrix[location_matrix[id_name] == id]
        subset_mtx = subset_mtx.drop(columns=[id_name])
        pairwise_distance = squareform(pdist(subset_mtx))
        adjacency_matrix = np.where(pairwise_distance > threshold, 0, pairwise_distance)
        neighbour_graph = nx.from_numpy_matrix(adjacency_matrix)
        flux = pd.DataFrame(neighbour_graph.edges, columns=["firstobjectnumber", "secondobjectnumber"])
        flux[id_name] = [id] * len(flux)
        flux = flux[flux.columns.tolist()[-1:] + flux.columns.tolist()[:-1]]
        relation.append(flux)
    relation = pd.concat(relation)
    relation.to_csv(output_csv, index=False)
