'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/18 14:53
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

x=torch.rand(1,784)

#in out
layer1=nn.Linear(784,200)
layer2=nn.Linear(200,200)
layer3=nn.Linear(200,10)

x=layer1(x)
x=F.relu(x,inplace=True)
print(x.shape)

x=layer2(x)
x=F.relu(x,inplace=True)
print(x.shape)

x=layer3(x)
x=F.relu(x,inplace=True)
print(x.shape)

'''
concisely
    inherit from nn.Model
    init layer in __init__
    implement forward()
'''
class MLP(nn.Module):

    def __init__(self):
        super(MLP,self).__init__()

        self.model=nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x=self.model(x)
        return x


