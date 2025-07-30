import torch
import torch.nn as nn

# 定义一个包含一个隐藏层的MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim,out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim,dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x=x.to(torch.float32)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x
