import torch
from script.prepare_data import read_and_generate_dataset


def get_PEMS08Dataset(device):
    graph_signal_matrix_filename = '../data/PEMS08/pems08.npz'
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

    return train_dataset, test_dataset
