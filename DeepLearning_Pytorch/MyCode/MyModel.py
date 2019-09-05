'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/1 14:59
'''
import torch
from torch import nn
from torch.nn import functional as F

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()

        self.liner=nn.Linear(1,1) #input and output is 1 dimension

    def forward(self, x):
        out=self.liner(x)
        return out

class Poly_model(nn.Module):
    def __init__(self):
        super(Poly_model,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 1),
        )

    def forward(self, x):
        out=self.model(x)
        return out

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x=self.model(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self,in_dim,n_hiddlen_1,n_hiddlen_2,out_dim):
        super(SimpleNet,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(in_dim,n_hiddlen_1),
            nn.BatchNorm1d(n_hiddlen_1),
            nn.ReLU(inplace=True),

            nn.Linear(n_hiddlen_1,n_hiddlen_2),
            nn.BatchNorm1d(n_hiddlen_2),
            nn.ReLU(inplace=True),

            nn.Linear(n_hiddlen_2,out_dim),
        )

    def forward(self,x):
        x=self.model(x)
        return x

class MNISTCnn(nn.Module):
    def __init__(self):
        super(MNISTCnn,self).__init__()#b,1,28,28
        self.model=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc=nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()#b,3,32,32

        layer1=nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1))
        #b,32,32,32
        layer1.add_module('relu1',nn.ReLU(inplace=True))
        layer1.add_module('pool1',nn.MaxPool2d(kernel_size=2,stride=2))#b,32,16,16
        self.layer1=layer1

        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1))
        #b,64,16,16
        layer2.add_module('relu2',nn.ReLU(inplace=True))
        layer2.add_module('pool2',nn.MaxPool2d(kernel_size=2,stride=2))#b,64,8,8
        self.layer2=layer2

        layer3=nn.Sequential()
        layer3.add_module('conv3',nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))
        #b,128,8,8
        layer3.add_module('relu3',nn.ReLU(inplace=True))
        layer3.add_module('pool3',nn.MaxPool2d(kernel_size=2,stride=2))#b,128,4,4
        self.layer3=layer3

        layer4=nn.Sequential()
        layer4.add_module('fc1',nn.Linear(2048,512))
        layer4.add_module('fc_relu1',nn.ReLU(inplace=True))
        layer4.add_module('fc2',nn.Linear(512,64))
        layer4.add_module('fc_relu2',nn.ReLU(inplace=True))
        layer4.add_module('fc3',nn.Linear(64,10))
        self.layer4=layer4

    def forward(self,x):
        conv1=self.layer1(x)
        conv2=self.layer2(conv1)
        conv3=self.layer3(conv2)
        fc_input=conv3.view(conv3.size(0),-1)
        fc_out=self.layer4(fc_input)
        return fc_out

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()#b,1,30,30

        layer1=nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(1,6,3,padding=1))#b,6,30,30
        layer1.add_module('pool1',nn.MaxPool2d(2,2))#b,6,15,15
        self.layer1=layer1

        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,5))#b,16,10,10
        layer2.add_module('conv2',nn.MaxPool2d(2,2))#b,16,5,5
        self.layer2=layer2

        layer3=nn.Sequential()
        layer3.add_module('fc1',nn.Linear(400,120))
        layer3.add_module('fc2',nn.Linear(120,84))
        layer3.add_module('fc3',nn.Linear(84,10))
        self.layer3=layer3

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(2)
        x=x.view(x.size(0),-1)
        x=self.layer3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self,num_classes):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self,num_classes):
        super(VGG,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )

        self._initialize_weights()

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels,eps=1e-3)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x,inplace=True)

class Inception(nn.Module):
    def __init__(self,in_channels,
                 n1x1, n3x3red, n3x3, n5x5red, n5x5,pool_features):
        super(Inception,self).__init__()
        # 1x1 conv branch
        self.branch1x1=BasicConv2d(in_channels,n1x1,kernel_size=1)
        # 1x1 conv -> 5x5 conv branch
        self.branch5x5_1=BasicConv2d(in_channels,n5x5red,kernel_size=1)
        self.branch5x5_2=BasicConv2d(n5x5red,n5x5,kernel_size=5,padding=2)

        # 1x1 conv -> 3x3 conv branch
        self.branch3x3dbl_1=BasicConv2d(in_channels,n3x3red,kernel_size=1)
        self.branch3x3dbl_2=BasicConv2d(n3x3red,n3x3,kernel_size=3,padding=1)
        self.branch3x3dbl_3=BasicConv2d(n3x3,n3x3,kernel_size=3,padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_1x1=BasicConv2d(in_channels,pool_features,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3dbl=self.branch3x3dbl_1(x)
        branch3x3dbl=self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl=self.branch3x3dbl_3(branch3x3dbl)

        branch_pool=self.branch_pool(x)
        branch_pool=self.branch_pool_1x1(branch_pool)

        outputs=[branch1x1,branch5x5,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.pre_layers = BasicConv2d(3, 192,
                                      kernel_size=3, padding=1)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def conv3x3(in_planes,out_planes,stride=1):
    '3x3 convolution with padding'
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

def conv1x1(in_planes,out_planes,stride=1):
    '1x1 convolution'
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        # 只有通道数翻倍的时候，空间分辨率才会缩小
        # 也就是只有每个大模块的第一次卷积会使用stide=2

        out+=residual
        out=self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)

        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual=self.downsample(x)

        out+=residual
        out=self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model