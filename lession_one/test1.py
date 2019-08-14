'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 15:28
'''
import torch
import numpy as np

x=torch.empty(5,3)
x=torch.rand(5,3)
x=torch.zeros(5,3,dtype=torch.long)

x=torch.tensor([5.5,3])

x=x.new_ones(5,3)

x=torch.rand_like(x,dtype=torch.double)

y=torch.rand(5,3,dtype=torch.double)
z=x+y
z=torch.add(x,y)

#in-place操作
#任何in-place的运算都会以_结尾，举例来说x.copy_(y),x.t_(),会改变x
y.add_(x)

#Resizing:  如果你希望resize/reshape一个tensor,可以使用torch.view
x=torch.rand(4,4)
y=x.view(16)
z=x.view(2,8)
#view中可以写一个-1，然后可以自动运算出来，不可以写两个-1，也不能写不能整除的数字
z=x.view(-1,8)
z=x.view(2,-1)


x=torch.rand(1)

#print(x.item())#拿取其中数字

'''
numpy 和 tensor 之间的转换
    在torch中 tensor和numpy array 之间相互转化非常容易
    torch tensor和numpy array会共享内存，所以改变其中一项也会改变另一项
    
'''

a=torch.ones(5)


#把tensor改变为numpy
b=a.numpy()
b[1]=2

#numpy转化成tensor 这样写均指向同一内存，ab都会改变
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)

#形如这样，a又重新指向了一个新的地址，不是原先的a
a=a+1
print(a)
print(b)

if torch.cuda.is_available():
    device=torch.device('cuda')
    y=torch.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to('cpu',torch.double))
