'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 21:20
'''
import torch
from torch import nn


# from torch.nn import functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积&池化
        self.conv_unit = nn.Sequential(
            # 卷积层，RGB3个channel，然后6个卷积核进行卷积，卷积核是5*5的，步长是1，没有zero padding
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 这里选择的是最大值池化，因为均值池化的效果很差，池化核是2*2的，步长是2，没有 zero padding
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 原理同上
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # 原理同上
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 全连接
        self.fc_unit = nn.Sequential(
            # 线性层，输入维度是16*5*5，输出维度是120,在图中有标识
            nn.Linear(16 * 5 * 5, 120),
            # 使用激活函数ReLU
            nn.ReLU(),
            # 第二个全连接层
            nn.Linear(120, 84),
            # 激活函数
            nn.ReLU(),
            # 输出层
            nn.Linear(84, 10)
        )
        '''
        tmp = torch.randn(2, 3, 32, 32)
        print('tmp: ',tmp.shape)
        out = self.conv_unit(tmp)
        print('conv out: ', out.shape)
        tmp=torch.randn(4,16*5*5)
        print('tmp: ', tmp.shape)
        out=self.fc_unit(tmp)
        print('fc out: ',out.shape)
        '''

    # 正向传播函数
    def forward(self, x):
        # 数据集batch的数量
        batch_size = x.size(0)
        # 卷积&池化操作
        x = self.conv_unit(x)
        # 通过view函数来调整x的大小
        x = x.view(batch_size, 16 * 5 * 5)
        # 全连接层操作
        logits = self.fc_unit(x)
        return logits


def main():
    # 实例化net模型
    net = LeNet5()
    # 初始化两张3通道的32*32的图像
    tmp = torch.randn(2, 3, 32, 32)
    #进行lenet操作流程
    out = net(tmp)
    print('LeNet out: ', out.shape)


if __name__ == '__main__':
    main()
