'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 20:46
'''
import torch
import torch.nn as nn


N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = nn.Sequential(
    nn.Linear(D_in, H, bias=False),  # w1*x+b
    nn.ReLU(),
    nn.Linear(H, D_out, bias=False)
)

nn.init.normal_(model[0].weight)
nn.init.normal_(model[2].weight)

loss_fn = nn.MSELoss(reduction='sum')

learing_rate = 1e-6
for t in range(500):
    # Forward pass
    y_pred = model(x)  # model.forward()

    # compute loss
    loss = loss_fn(y_pred, y)  # computation graph
    print(t, loss.item())

    # backward pass
    # compute the gradient
    loss.backward()

    # update weights of W1 and W2
    with torch.no_grad():
        for param in model.parameters():  # param {tensor,grad}
            param -= learing_rate * param.grad

    model.zero_grad()
