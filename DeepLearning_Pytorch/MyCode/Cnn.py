'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/3 15:03
'''
import torch
from torch import nn,optim
from MyModel import SimpleCNN,Lenet,AlexNet,GoogLeNet,Bottleneck

'''
model=SimpleCNN()
# print(model)
# print(model.children())
# print(model.modules())
#
# new_model=nn.Sequential(*list(model.children())[:1])
#print(new_model)
# print(model.named_children())
# print(model.named_modules())
conv_modle=nn.Sequential()
for layer in model.named_modules():
    print(layer[0])
    if isinstance(layer[1],nn.Conv2d):
        print('true')
        conv_modle.add_module(layer[0],layer[1])

print(conv_modle)
'''


model=Bottleneck(32,32)
print(model.expansion)

