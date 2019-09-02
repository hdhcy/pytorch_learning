'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/1 14:57
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from MyModel import LinearRegression, Poly_model



def OneLinear():

    # 读入数据 x 和 y
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    plt.figure()
    plt.scatter(x_train,y_train,c='r')

    if torch.cuda.is_available():
        model=LinearRegression().cuda()
    else:
        model=LinearRegression()
    criterion=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=1e-3)

    num_epochs=1000
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            inputs=torch.tensor(x_train).cuda()
            target=torch.tensor(y_train).cuda()
        else:
            inputs=torch.tensor(x_train)
            target=torch.tensor(y_train)

        #forward
        out=model(inputs)
        loss=criterion(out,target)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1)%20==0:
            print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,num_epochs,loss.item()))
    model.eval()
    predict=model(inputs)
    predict=predict.data.numpy()
    plt.plot(x_train,predict,'b')
    plt.show()

def make_features(x):
    x=x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

def f(x):
    w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
    b_target = torch.FloatTensor([0.9])
    return x.mm(w_target)+b_target[0]

def get_batch(batch_size=32):
    random=torch.randn(batch_size)
    x=make_features(random)
    y=f(x)

    return x,y



def MulLinear():
    model=Poly_model()
    criterion=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=1e-3)

    epoch=0
    while True:
        batch_x,batch_y=get_batch()
        #forward
        output=model(batch_x)
        loss=criterion(output,batch_y)
        #reset gradients
        optimizer.zero_grad()
        #backward
        loss.backward()
        #update parameters
        optimizer.step()
        epoch+=1
        print(loss.item())
        if loss.item()<1e-3:
            break
    model.eval()
    print()



def main():
    #OneLinear()
    MulLinear()


if __name__ == '__main__':
    main()