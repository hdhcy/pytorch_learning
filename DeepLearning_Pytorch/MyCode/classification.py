'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/1 17:41
'''
import torch
from torch import nn,optim
import numpy as np
import matplotlib.pyplot as plt
from MyModel import LogisticRegression

def Clssfication():
    with open('data.txt','r') as f:
        data_list=f.readlines()
        data_list=[i.split('\n')[0] for i in data_list]
        data_list=[i.split(',') for i in data_list]
        data=[(float(i[0]),float(i[1]),float(i[2])) for i in data_list]

    x0=list(filter(lambda x:x[-1]==0.0,data))
    x1=list(filter(lambda x:x[-1]==1.0,data))


    plot_x0_0=[i[0] for i in x0]
    plot_x0_1=[i[1] for i in x0]
    plot_x1_0=[i[0] for i in x1]
    plot_x1_1=[i[1] for i in x1]

    plt.plot(plot_x0_0,plot_x0_1,'ro',label='x_0')
    plt.plot(plot_x1_0,plot_x1_1,'bo',label='x_1')
    #plt.show()

    x_data=[i[0:2] for i in data]
    y_data=[i[-1] for i in data]

    logistic_model=LogisticRegression()
    criterion=nn.BCELoss()
    optimizer=optim.SGD(logistic_model.parameters(),lr=1e-3,momentum=0.9)

    # x=torch.tensor(x_data)
    # y=torch.tensor(y_data).unsqueeze(1)
    # print(x.shape)
    # print(y.shape)
    # w0,w1=logistic_model.model[0].weight[0]
    # print(w0.item())
    # print(w1.item())
    # print(logistic_model.model[0].bias.item())
    x = torch.tensor(x_data)
    y = torch.tensor(y_data).unsqueeze(1)

    epoch_num=50000
    for epoch in range(epoch_num):

        #forward
        out=logistic_model(x)
        loss=criterion(out,y)
        mask=out.ge(0.5).float()
        correct=(mask==y).sum().float()
        acc=correct/x.size(0)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch+1)%10000==0:
            print('*'*10)
            print('epoch {}'.format(epoch+1))
            print('loss is {:.4f}'.format(loss.item()))
            print('acc is {:.4f}'.format(acc))

    #w1x+w2y+b=0
    w0,w1=logistic_model.model[0].weight[0]
    w0=w0.item()
    w1=w1.item()
    b=logistic_model.model[0].bias.item()
    plot_x=np.arange(30,100,0.1)
    plot_y=(-w0*plot_x-b)/w1
    plt.plot(plot_x,plot_y)
    plt.show()

def main():
    Clssfication()

if __name__ == '__main__':
    main()


