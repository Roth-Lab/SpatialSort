import pandas as pd
from src.preprocessing.DataCube import DataCube
import networkx as nx


def read_data(matrix_csv, location_csv, neighbour_csv):
    """
        Returns a list of patient data cube
    """
    # Reading csvs if not already data frame
    if type(matrix_csv) is not pd.DataFrame:
        read_matrix = pd.read_csv(matrix_csv)
        location_list = pd.read_csv(location_csv)
        relation_list = pd.read_csv(neighbour_csv)
    else:
        read_matrix = matrix_csv
        location_list = location_csv
        relation_list = neighbour_csv

    # Obtain unique ids
    col = list(read_matrix.columns)
    id_name = col[0]
    col.remove(id_name)
    unique_ids = pd.unique(read_matrix[id_name])

    # Returning cube list
    patient_list = []

    # Loop through images
    for image_id in unique_ids:
        # Slice data frames to processes
        expression_matrix = read_matrix[read_matrix[id_name] == image_id]
        location_matrix = location_list[location_list[id_name] == image_id]
        relation = relation_list[relation_list[id_name] == image_id]

        # Drop file_id to place into data cubes
        expression_matrix = expression_matrix.drop(columns=[id_name])
        location_matrix = location_matrix.drop(columns=[id_name])
        relation = relation.drop(columns=[id_name])

        # Construct graph
        neighbor_graph = nx.Graph()
        tuples = [tuple(x) for x in relation.values]

        # Adding in self-relations
        singletons = list(set(range(location_matrix.shape[0])) - set(
            list(relation.iloc[:, 0].values) + list(relation.iloc[:, 1].values)))
        for single in singletons:
            tuples.append((single, single))

        neighbor_graph.add_edges_from(tuples)

        labels = [0] * location_matrix.shape[0]
        nx.set_node_attributes(neighbor_graph, dict(enumerate(labels)), 'label')

        # Create data cube object
        cube = DataCube(expression_mtx=expression_matrix.to_numpy(),
                        location_mtx=location_matrix.to_numpy(),
                        relation=relation.to_numpy())
        cube.id = image_id
        cube.gene_names = col
        cube.neighbor_graph = neighbor_graph
        patient_list.append(cube)

    return patient_list
