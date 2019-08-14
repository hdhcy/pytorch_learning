'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 19:59
'''
import torch

N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learing_rate = 1e-6
for t in range(500):
    # Forward pass
    h = x.mm(w1)  # N*H
    h_relu = h.clamp(min=0)  # N*H
    y_pred = h_relu.mm(w2)  # N*D_out

    # compute loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.T)
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update weights of W1 and W2
    w1 -= learing_rate * grad_w1
    w2 -= learing_rate * grad_w2
