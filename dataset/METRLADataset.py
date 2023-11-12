import numpy as np
import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def load_METRLADataset():
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    print("Dataset type:  ", dataset)
    print("Number of samples sequences: ",  len(list(dataset)))
    return dataset

def split_METRLADataset():
    # from torch.utils.data import DataLoader
    dataset = load_METRLADataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)

    print("Number of train buckets: ", len(list(train_dataset)))
    # print(list(train_dataset))
    print("Number of test buckets: ", len(list(test_dataset)))
    print("Type of train_dataset",[train_dataset])
    return train_dataset,test_dataset

def get_METRLADataset(device):
    train_dataset,test_dataset = split_METRLADataset()
    train_input = np.array(train_dataset.features)  # (27399, 207, 2, 12)
    train_target = np.array(train_dataset.targets)  # (27399, 207, 12)
    train_x_tensor = torch.from_numpy(train_input[:,:,0,:]).type(torch.FloatTensor).to(device).unsqueeze(2)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    test_input = np.array(test_dataset.features)  # (, 207, 2, 12)
    test_target = np.array(test_dataset.targets)  # (, 207, 12)
    test_x_tensor = torch.from_numpy(test_input[:,:,0,:]).type(torch.FloatTensor).to(device).unsqueeze(2) # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)  # (B, N, T)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    return train_dataset_new,test_dataset_new

def get_METRLA_adjacent_matrix(device):
    train_dataset, _ = split_METRLADataset()
    edge_index = torch.from_numpy(np.asarray(train_dataset[0].edge_index)).type(torch.FloatTensor).to(device)
    return edge_index

def get_METRLA_2Dadjacent_matrix(device):
    train_dataset, _ = split_METRLADataset()
    edge_index = np.asarray(train_dataset[0].edge_index)
    adj_matrix = torch.zeros(207,207)
    for i in range(207):
        adj_matrix[edge_index[0],edge_index[1]] = 1
    adj_matrix = torch.from_numpy(np.asarray(adj_matrix)).type(torch.FloatTensor).to(device)
    return adj_matrix

