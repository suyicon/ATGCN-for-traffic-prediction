import csv

import numpy as np
import torch
from script.prepare_data import read_and_generate_dataset

def get_PEMS04Dataset(device):
    graph_signal_matrix_filename = '../data/PEMS04/pems04.npz'
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks=1,
                                         num_of_days=1,
                                         num_of_hours=1,
                                         num_for_predict=12,
                                         points_per_hour=12,
                                         merge=True)
    train_tensor_x = torch.from_numpy(all_data['train']['recent']).type(torch.FloatTensor).to(device)
    train_tensor_y = torch.from_numpy(all_data['train']['target']).type(torch.FloatTensor).to(device)
    test_tensor_x = torch.from_numpy(all_data['test']['recent']).type(torch.FloatTensor).to(device)
    test_tensor_y = torch.from_numpy(all_data['test']['target']).type(torch.FloatTensor).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)

    return train_dataset,test_dataset

def get_PEMS04_adjacency_matrix(num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    distance_df_filename='../data/PEMS04/distance.csv'
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)

    for i, j, distance in edges:
        A[i, j] = 1
        distaneA[i, j] = distance

    return A, distaneA






