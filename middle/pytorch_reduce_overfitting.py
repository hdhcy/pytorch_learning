'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/26 21:51
'''
'''
Reduce Overfitting
    More data
    Constraint model complexity
        shallow
        regularization
    Dropout
    Data argumentation
    Early Stopping
'''

#Regularization

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from visdom import Visdom

batch_size = 200
learning_rate = 0.01
epochs = 10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 400),
            nn.LeakyReLU(inplace=True),
            nn.Linear(400, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

viz=Visdom()
viz.line([0.0],[0.0],win='train_loss',opts=dict(title='train loss'))
viz.line([[0.0,0.0]],[0.0],win='test',opts=dict(title='test loss and acc',
                                               legend=['loss','acc']))
#L2-regularization 设置weight_decay 迫使||w||->0
net = MLP()
optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.01)
criteon = nn.CrossEntropyLoss()

'''
#L1-regularization
regularization_loss=0
for param in net.parameters():
    regularization_loss+=torch.sum(torch.abs(param))

classify_loss=criteon(logits,target)
loss=classify_loss+0.01*regularization_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
'''

global_step=0
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        global_step+=1
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        viz.line([loss.item()],[global_step],win='train_loss',update='append')

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    viz.line([[test_loss,correct/len(test_loader.dataset)]],[global_step],win='test',update='append')
    viz.images(data.view(-1,1,28,28),win='x')
    viz.text(str(pred.numpy()), win='pred', opts=dict(title='pred'))

