import torch
import torch.nn.functional as F
from tqdm import tqdm
from STTNmodel.STTN import STTN
import dataset.METRLADataset as dataset

train_loss=[]
test_loss=[]
predictions = []
labels = []

class TemporalGNN(torch.nn.Module):
    def __init__(self, periods,adj):
        super().__init__()
        self.sttn = STTN(n_nodes=207, len_his=periods, len_pred=periods,adj=adj)
    def forward(self, x):
        h = self.sttn(x)
        return h

def make_model(device,adj):
    net = TemporalGNN(periods=12,adj=adj)
    net.to(device)
    print(net)
    return net

def train(train_loader, test_loader, model, optimizer,device):
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
            # print("x la", x.shape)
            # print("x lal", x[0, :, :])
            # print("x lal",x[0,:,0])
            y_hat = model(x).squeeze(0).squeeze(-1)
            # print("y_hat",y_hat.shape)
            # print("y",y.shape)
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
            test(test_loader, device, model)


def test(test_loader, device, model):
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
                y_hat = model(x)
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