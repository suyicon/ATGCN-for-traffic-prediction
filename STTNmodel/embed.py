from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(Embedding, self).__init__()
        self.conv = nn.Conv2d(d_feature, d_model, (1, 1))

    def forward(self, x):
        x = x.unsqueeze(0)
        # print(x.shape)
        # 批次，传感器数量，特征，时间步
        x = x.permute(0,2, 1, 3)#批次，特征，传感器，时间步
        x = self.conv(x)
        x = x.permute(0,3, 2, 1)#批次，时间步，传感器，特征
        return x
