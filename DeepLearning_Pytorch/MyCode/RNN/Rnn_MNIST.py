'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/6 15:58
'''
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from Model import Rnn
import matplotlib.pyplot as plt
from visdom import Visdom

def trainMnist():
    #超参数(Hyperparameters)
    batch_size=128
    learning_rate=1e-3
    num_epoches=20

    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_tf=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])]
    )

    #下载训练集MNIST手写数字训练集
    train_dataset=datasets.MNIST(
        root='../data',train=True,transform=data_tf,download=True
    )

    test_dataset=datasets.MNIST(
        root='../data',train=False,transform=data_tf
    )
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    viz = Visdom()
    viz.line([[0.0, 0.0]], [0.0], win='train', opts=dict(title='train_loss and train_acc',
                                                                          legend=['train_loss', 'train_acc']))
    viz.line([0.0], [0.0], win='test', opts=dict(title='test loss and acc'))
    #model=SimpleNet(28*28,300,100,10)
    net=Rnn()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)

    best_acc=80
    with open("MINST_acc.txt", "w") as f:
        with open("MINST_log.txt", "w")as f2:
            for epoch in range(num_epoches):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    viz.line([[1. * float(correct) / total,loss.item()]], [i + 1 + epoch * length], win='train', update='append')
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * float(correct) / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * float(correct) / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        #viz.images(images.view(-1, 1, 28, 28), win='x', )
                        #viz.text(str(predicted.numpy()), win='pred', opts=dict(title='pred'))

                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    viz.line([1. * correct / total],[epoch+1],win='test',update='append')

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("MINST_best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % num_epoches)

def main():
    trainMnist()

if __name__ == '__main__':
    main()

