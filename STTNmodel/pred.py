from torch import nn


class Pred(nn.Module):
    def __init__(self, d_feature, d_model, len_his, len_pred):
        super(Pred, self).__init__()
        self.conv1 = nn.Conv2d(len_his, len_pred, (1, 1))
        self.conv2 = nn.Conv2d(d_model, d_feature, (1, 1))
        self.relu = nn.ReLU()
        #self.linear = nn.Linear(in_features=d_feature,out_features=)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = x.permute(0,3, 1, 2)
        #x = x.permute(2,0,1)
        x = self.conv2(x)
        x = x.permute(0,3, 2, 1)
        #x = x.permute(1,2,0)
        return x
