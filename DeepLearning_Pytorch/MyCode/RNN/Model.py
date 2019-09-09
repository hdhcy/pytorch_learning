'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/6 15:51
'''
import torch
from torch import nn,optim

class Rnn(nn.Module):
    def __init__(self,in_dim=28,hidden_dim=100,n_class=10,num_layers=2):
        super(Rnn,self).__init__()
        self.lstm=nn.LSTM(in_dim,hidden_dim,num_layers,batch_first=True)
        self.classifier=nn.Linear(hidden_dim,n_class)

    def forward(self,x):
        '''
        x 大小为 (batch, 1, 28, 28)，所以我们需要将其转换成 RNN 的输入形式，即 (28, batch, 28)
        '''
        x = x.squeeze()# 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        out,_=self.lstm(x)
        out=out[:,-1,:]# 取序列中的最后一个，大小是 (batch, hidden_feature)
        out=self.classifier(out)
        return out




