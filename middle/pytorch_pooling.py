'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 10:20
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.rand(1,16,18,18)

#window_size=2 stride=2
layer=nn.MaxPool2d(2,stride=2)

out=layer(x)
print(out.shape)

out=F.avg_pool2d(x,2,stride=2)
print(out.shape)

#upsample 放大
x=out
out=F.interpolate(x,scale_factor=2,mode='nearest')
print(out.shape)

out=F.interpolate(x,scale_factor=3,mode='nearest')
print(out.shape)

#One unit: conv2d-Biary Normization-Pooling-Relu
print(x.shape)
layer=nn.ReLU(inplace=True)
out=layer(x)
print(out.shape)

out=F.relu(x)
print(out.shape)
