import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device('cuda')
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, last=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.last = last

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

    def forward(self, X, A):
        #print(A.shape)
        #B,M,_,_ = X.shape
        A_tilde = A + torch.eye(A.shape[0]).to(DEVICE)
        #print(A_tilde.shape)
        D = A.sum(-1).to(DEVICE)
        #print(D)
        D = torch.pow(D,-0.5).to(DEVICE)
        #print(D[...,:].shape)
        D_tilde = torch.diag(D).to(DEVICE)
        #print(D_tilde)
        #print(D_tilde.shape)
        DA = torch.matmul(D_tilde, A_tilde).to(DEVICE)
        A_aggregation = torch.matmul(DA,D_tilde).to(DEVICE)
        AX = torch.matmul(A_aggregation, X).to(DEVICE)
        out = self.W(AX)

        if self.last:
            return out
        else:
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out


class GCN(nn.Module):
    def __init__(self, C, F, H, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(C, H, dropout, False)
        self.conv2 = GCNLayer(H, F, dropout, True)
        self.dropout = dropout
    def forward(self, X, A):
        hidden = self.conv1(X, A)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden,self.dropout,training=self.training)
        out = self.conv2(hidden, A)
        return out