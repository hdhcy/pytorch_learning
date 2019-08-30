'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/27 20:39
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


#input kernelnumber stride padding
layer=nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0)
x=torch.rand(4,3,28,28)
out=layer.forward(x)
print(out.shape)


'''

layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
x=torch.rand(1,1,28,28)
out=layer.forward(x)
print(out.shape)

layer=nn.Conv2d(4,3,kernel_size=3,stride=2,padding=1)
x=torch.rand(1,4,28,28)
out=layer.forward(x)
print(out.shape)

#推荐这样调用
out=layer(x)
print(out.shape)
#weight [kernel_number,input_number,kernel_size,kernelsize]
print(layer.weight.shape)
print(layer.bias.shape)


#pytorch中的一个比较low的
x=torch.randn(4,3,28,28)
w=torch.rand(16,3,5,5)
b=torch.rand(16)

out=F.conv2d(x,w,b,stride=1,padding=1)
print(out.shape)

out=F.conv2d(x,w,b,stride=2,padding=2)
print(out.shape)
'''

