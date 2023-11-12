import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.lstm_gat_gcn import LSTMGATGCN

train_loss=[]
test_loss=[]
predictions = []
labels = []

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super().__init__()
        self.tgnn = LSTMGATGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        self.linear = torch.nn.Linear(32, periods)
    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

def make_model(device):
    net = TemporalGNN(node_features=2, periods=12)
    net.to(device)
    print(net)
    return net

def train(train_loader, test_loader, model, optimizer, edge_index,device):
    # set train mode
    model.train()
    print("Running training on ", device)
    # train
    for batch, (X, Y) in tqdm(enumerate(train_loader)):
        loss = 0
        step = 0
        # print(data)
        # forward
        for x, y in zip(X, Y):
            # print(snapshot)
            y_hat = model(x, edge_index)
            l = torch.nn.MSELoss()
            loss = loss + l(y_hat, y)
            step += 1
        # backward
        loss = loss / (step + 1)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        if (batch + 1) % 10 == 0:
            print("average Batch {} train MSE: {:.4f}".format(batch, loss.item()))
            test(test_loader, device, model, edge_index)


def test(test_loader, device, model, edge_index):
    model.eval()
    total_loss = 0
    num_batches = len(test_loader)
    with torch.no_grad():
        for (X, Y) in test_loader:
            loss = 0
            step = 0
            # test
            for x, y in zip(X, Y):
                # compare label and prediction
                y_hat = model(x, edge_index)
                loss = loss + torch.mean((y_hat - y) ** 2)
                labels.append(y)
                predictions.append(y_hat)
            loss = loss / (step + 1)
            loss = loss.item()
            total_loss += loss
        # caculate average loss
        avg_loss = total_loss / num_batches
        print("average Test MSE: {:.4f}".format(avg_loss))
        test_loss.append(avg_loss)