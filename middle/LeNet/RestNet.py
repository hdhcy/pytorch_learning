'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 21:41
'''
import torch
from torch import nn
from torch.nn import functional as F


# 定义resnet网络模型的block
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        # 对输入的数据进行卷积操作，卷积核是3*3，步长是1，有长为1的padding
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        # 进行batch normalization 操作，就是将数据分布批量规范化，方便后续的训练
        self.bn1 = nn.BatchNorm2d(ch_out)

        # 将上一层的输出作为这一层的输入进行训练，同样卷积核是3*3，步长是1，四周加了一层padding
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        # 将第二层的数据进行batch normalization操作
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 需要保持输入和输出的size要相同，否则就加一个单元让输入和输出的size保持相同
        self.extra = nn.Sequential()
        # [b, ch_in, h, w] =>[b, ch_out, h, w]
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        #print('X :',x.shape)
        # 第一层卷积和batch norm操作，加一层ReLU激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        #print('out1:',out.shape)
        # 第二层卷积和batch norm操作
        out = self.bn2(self.conv2(out))
        #print('out2:', out.shape)
        # 将shortcut输出的x和前面block卷积操作的结果进行叠加
        out = self.extra(x) + out
        #print('x:',self.extra(x).shape)
        #print('out3:', out.shape)

        return out


# 为了减轻CPU/GPU负担，我将输入和输出维度以及block的个数设置得很小，其实用GPU的话，可以适当增加维度和block的个数
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 定义卷积操作
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        # followed 4 blocks，这里我只用两个block，减轻CPU的负担
        self.blk1 = ResBlk(16, 16)
        self.blk2 = ResBlk(16, 32)
        # self.blk3 = ResBlk(32, 64)
        # self.blk4 = ResBlk(64, 128)
        # 定义输出层，将维度降到10输出，完成图像分类
        self.outlayer = nn.Linear(32 * 32 * 32, 10)

    # 进行前向传播操作
    def forward(self, x):
        """
        :param x:
        :return:
        """
        # 卷积核block级联
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)

        # 检查x的数据size并做调整
        x = x.view(x.size(0), -1)
        # 输出结果
        x = self.outlayer(x)

        return x


def main():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('blk: ', out.shape)

    model = ResNet()
    tmp = torch.randn(2, 3, 32, 32)
    out = model(tmp)
    print('resnet: ', out.shape)


if __name__ == '__main__':
    main()
