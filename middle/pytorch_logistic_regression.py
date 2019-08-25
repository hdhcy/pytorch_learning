'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/17 14:35
'''
import torch
from torch.nn import functional as F

#Cross Entropy Loss
#Entropy
a=torch.full([4],1/4)
print(a)

entropy=-(a*a.log2()).sum()
print(entropy)

a=torch.tensor([0.1,0.1,0.1,0.7])
entropy=-(a*a.log2()).sum()
print(entropy)

a=torch.tensor([0.001,0.001,0.001,0.997])
entropy=-(a*a.log2()).sum()
print(entropy)

#Numerical Stability
x=torch.randn(1,784)
w=torch.randn(10,784)

logits=x@w.t()
print(logits.shape)

pred=F.softmax(logits,dim=1)
print(pred.shape)

pred_log=torch.log(pred)

#cross_entropy=softmax+log+null_loss
#使用cross_entropy必须传入logits,在pytorch中cross_entropy已经把log和softmax函数打包在一起了
print(F.cross_entropy(logits,torch.tensor([3])))

#不使用cross_entropy自己这样计算
print(F.nll_loss(pred_log,torch.tensor([3])))



