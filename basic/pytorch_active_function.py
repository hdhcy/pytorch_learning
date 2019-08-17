'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/16 15:19
'''
import torch

#sigmoid
a=torch.linspace(-100,100,11)
print(a)

b=a.sigmoid()
print(b)

#tanh
a=torch.linspace(-1,1,10)
print(a)

b=a.tanh()
print(b)

from torch.nn import functional as F

#relu
a=torch.linspace(-1,1,10)

b=torch.relu(a)
c=a.relu()
d=F.relu(a)

print(b,c,d)

#softmax
a=torch.rand(3,requires_grad=True)
print(a)

p=F.softmax(a,dim=0)

#p1对[a]进行求导 返回p1对a的grad i和j相等时求导是正的
print(torch.autograd.grad(p[1],[a],retain_graph=True))