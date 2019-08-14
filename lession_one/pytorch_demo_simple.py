'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 20:11
'''
import torch

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b  # y=2*1+3

y.backward()

'''
#dy/dw=x
print(w.grad)

print(x.grad)
print(b.grad)
'''

N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learing_rate = 1e-6
for t in range(500):
    # Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute loss
    loss = (y_pred - y).pow(2).sum()  # computation graph
    print(t, loss.item())

    # backward pass
    # compute the gradient
    loss.backward()

    # update weights of W1 and W2
    with torch.no_grad():
        w1 -= learing_rate * w1.grad
        w2 -= learing_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
