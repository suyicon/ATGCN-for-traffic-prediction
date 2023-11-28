import torch
import torch.nn.functional as F
from models.lstm_gat_gcn import LSTMGATGCN
from tqdm import tqdm
import torch.nn
from trainer.STTN_train import make_model as make_model
from trainer.STTN_train import train as train
from dataset.METRLADataset import get_METRLADataset as get_dataset
from dataset.METRLADataset import get_METRLA_2Dadjacent_matrix as get_adjacent_matrix
from trainer.STTN_train import train_loss,test_loss

DEVICE = torch.device('cuda')
adj = get_adjacent_matrix(DEVICE)
model = make_model(DEVICE,adj)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
batch_size = 32
epochs = 20
train_dataset, test_dataset = get_dataset(DEVICE)
print("tr",train_dataset[0,0,0])
#edge_index = get_adjacent_matrix(DEVICE)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    #train(train_loader, test_loader, model, optimizer, edge_index, DEVICE)
    train(train_loader, test_loader, model, optimizer, DEVICE)
torch.save(model, "lstgatgcn_pth")
torch.save(model.state_dict, "lstgatgcn_w.pth")

import matplotlib.pyplot as plt
batch_num_train = range(0,len(train_loss))
batch_num_test = range(0,len(test_loss)*10,10)

plt.plot(batch_num_train,train_loss , '-', linewidth=1.0, color='g')
plt.plot(batch_num_test,test_loss , '-', linewidth=1.0, color='b')
plt.legend(['train_loss','test_loss'], fontsize=10)
plt.show()