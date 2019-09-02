'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/1 14:59
'''
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()

        self.liner=nn.Linear(1,1) #input and output is 1 dimension

    def forward(self, x):
        out=self.liner(x)
        return out

class Poly_model(nn.Module):
    def __init__(self):
        super(Poly_model,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 1),
        )

    def forward(self, x):
        out=self.model(x)
        return out

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x=self.model(x)
        return x


