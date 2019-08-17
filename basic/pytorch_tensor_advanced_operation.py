'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/16 13:26
'''
import torch

#where
cond=torch.rand(2,2)
print(cond)

a=torch.zeros(2,2)
b=torch.ones(2,2)

c=torch.where(cond>0.5,a,b)
print(c)

#gather
'''
retrieve global label
    argmax (pred) to get relative labeling
    On some condition,our lable is dinstinct from relative labeling
'''
prod=torch.randn(4,10)

idx=prod.topk(dim=1,k=3)
print(idx)

idx=idx[1]
print(idx)

label=torch.arange(10)+100
print(label)

print(idx.long())
res=torch.gather(label.expand(4,10),dim=1,index=idx.long())
print(res)