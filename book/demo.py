from torch.autograd import Variable
import torch as t

x=Variable(t.ones(2,2),requires_grad=True)

y=x.sum()
print(y)