'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/16 20:52
'''
import torch
from torch.nn import functional as F

x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=True)

o=torch.sigmoid(x@w.t())
print(o.shape)

loss=F.mse_loss(torch.ones(1,2),o)
print(loss)

loss.backward()
print(w.grad)
print(w.grad.shape)