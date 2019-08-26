'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/26 14:31
'''
import torch
from torch.nn import functional as F

logits=torch.rand(4,10)

pred=F.softmax(logits,dim=1)
print(pred.shape)

pred_label=pred.argmax(dim=1)
print(pred_label)

print(logits.argmax(dim=1))

label=torch.tensor([9,3,2,4])
correct=torch.eq(pred_label,label)
print(correct)

accuracy=correct.sum().float().item()/4
print(accuracy)

'''
When to test
    test once per several batch
    test once per epoch
    epoch V.S. step?
'''