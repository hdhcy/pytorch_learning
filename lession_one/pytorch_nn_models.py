'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 21:29
'''

import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


class TwoLayerNet(nn.Module):

    # define the model architecture
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H, bias=False)
        self.linear2 = nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwoLayerNet(D_in, H, D_out)

loss_fn = nn.MSELoss(reduction='sum')

learing_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

for t in range(500):
    # Forward pass
    y_pred = model(x)  # model.forward()

    # compute loss
    loss = loss_fn(y_pred, y)  # computation graph
    print(t, loss.item())

    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update model parameters
    optimizer.step()
