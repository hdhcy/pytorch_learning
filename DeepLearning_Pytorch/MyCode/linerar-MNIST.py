'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/2 15:52
'''
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from MyModel import SimpleNet,MNISTCnn
import matplotlib.pyplot as plt

def trainMnist():
    #超参数(Hyperparameters)
    batch_size=128
    learning_rate=1e-2
    num_epoches=20

    data_tf=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])]
    )

    #下载训练集MNIST手写数字训练集
    train_dataset=datasets.MNIST(
        root='./data',train=True,transform=data_tf,download=True
    )

    test_dataset=datasets.MNIST(
        root='./data',train=False,transform=data_tf
    )
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    a_img,a_lable=train_dataset[0]
    a_img=a_img.view(28,28)
    plt.imshow(a_img)
    #plt.show()
    print(a_img.shape)
    print(a_lable)

    #model=SimpleNet(28*28,300,100,10)
    model=MNISTCnn()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epoches):
        model.train()
        for batch_idx,(data,target) in enumerate(train_loader):
            #print(data.shape)
            #data=data.view(-1,28*28)

            logits=model(data)
            loss=criterion(logits,target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        test_loss = 0
        correct = 0
        model.eval()
        for data, target in test_loader:
            #data = data.view(-1, 28 * 28)
            logits = model(data)
            test_loss += criterion(logits, target).item()

            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    trainMnist()

if __name__ == '__main__':
    main()
