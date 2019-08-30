'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 21:39
'''
# 导入必要的python库
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from LeNet5 import LeNet5
from torch import nn, optim
import torch
from RestNet import ResNet


def main():
    # 初始化batch size为32
    batch_size = 32

    # 定义cifar10训练集，确保训练集数据大小是32*32
    cifar_train = datasets.CIFAR10(
        '../cifar',
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.ToTensor()
            ]),
        download=True)

    # 使用数据加载器加载训练集，并随机打乱
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    # 定义测试集
    cifar_test = datasets.CIFAR10(
        '../cifar',
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(32,32),
                transforms.ToTensor()
            ]),
        download=True)
    # 使用数据加载器加载测试集
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)
    # device = torch.device('cuda')
    # model = LeNet5().to(device)
    # 初始化LeNet5模型
    #model = LeNet5()
    model = ResNet()
    # 模型优化的标准是交叉熵
    criteon = nn.CrossEntropyLoss()
    # 使用Adam进行优化，加载模型的参数，设置学习率为0.001
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    # 循环100个epoch
    for epoch in range(100):
        # 将 module 设置为 training mode
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            logits = model(x)
            loss = criteon(logits, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:',epoch,'  Loss:', loss.item())
        # 将模型设置成 evaluation
        model.eval()
        # 表示下面的部分不需要求梯度
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                logits = model(x)
                # 预测的类别
                pred = logits.argmax(dim=1)
                # 总的正确分类数
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            # 正确率
            acc = total_correct / total_num
            print('Epoch:',epoch, '  acc: ', acc)


if __name__ == '__main__':
    main()
