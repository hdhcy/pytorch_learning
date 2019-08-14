'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/13 10:44
'''
import numpy as np
import torch

NUM_DIGITS=10

def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_dencode(i, prediction):
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][prediction]

#十进制转化成二进制
def binary_enocode(i,num_digits):
    return np.array([i>>d &1 for d in range(num_digits)][::-1])

trX=torch.Tensor([binary_enocode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY=torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])


NUM_HIDDEN=100
model=torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN,4)#4 logits,after sofmax,we get a probablity distribution
)

loss_fu=torch.nn.CrossEntropyLoss()
optimiter=torch.optim.SGD(model.parameters(),lr=0.05)

BATCH_SIZE=128
for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchX=trX[start:end]
        batchy=trY[start:end]

        y_pred=model(batchX)#forward

        loss=loss_fu(y_pred,batchX)
        print('Epoch',epoch,loss.item())

        optimiter.zero_grad()
        loss.backward()#backpass
        optimiter.step()#gradient descent
