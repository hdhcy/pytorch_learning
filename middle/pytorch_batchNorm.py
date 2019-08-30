'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 10:49
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Feature scaling
    Image Normalization
        normalization=transforms.Normalize(mean=[0.485,0.456,0.406],
                                           std=[0.229,0.224,0.225])
    Batch Normalization
        Batch Norm
        Layer Norm
        Instance Norm
        Group Norm

        Advantages
            Converge faster
            Better performance
            Robust
                stable
                larger learning rate
'''
x = torch.rand(100, 16, 28, 28)

# chanel的数量
layer = nn.BatchNorm2d(16)
out = layer(x)
# u
print(layer.running_mean)
# sig
print(layer.running_var)


x = torch.rand(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print(out.shape)

print(layer.weight)
print(layer.weight.shape)
print(layer.bias)
print(layer.bias.shape)
print(layer.running_mean)
print(layer.running_var)
print(vars(layer))
layer.eval()
print(vars(layer))
print(layer.eval())
print(vars(layer))
