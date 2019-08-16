'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/15 10:46
'''
import torch

# basic operations
a = torch.rand(3, 4)
b = torch.rand(4)
print(a, b)

print(a + b)
print(a - b)
print(a * b)
print(b * a)
print(a / b)

'''
+:  add
-:  sub
*:  mul
/:  div
'''
flag = torch.all(torch.eq(a + b, torch.add(a, b)))
print(flag)
flag = torch.all(torch.eq(a - b, torch.sub(a, b)))
print(flag)
flag = torch.all(torch.eq(a * b, torch.mul(a, b)))
print(flag)
flag = torch.all(torch.eq(a / b, torch.div(a, b)))
print(flag)

# matuml
'''
torch.mm
    only for 2d
torch.matmul
@
'''
a = torch.ones(2, 2) * 3
b = torch.ones(2, 2)

c = torch.mm(a, b)
print(c)

# 推荐这种方式
d = torch.matmul(a, b)
print(d)

e = a@b
print(e)

# an example [4,784]=>[4,512] [4,784]*[784,512]
x = torch.rand(4, 784)
# pytorch 的写法是第一个数字为降维的维度512，第二个数字为进来的维度784
w = torch.rand(512, 784)

y = torch.matmul(x, w.t())
print(y.shape)

# >2d tensor matnul?
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)

c = torch.matmul(a, b)
print(c.shape)

# 适合boradcasting
b = torch.rand(4, 1, 64, 32)
c = torch.matmul(a, b)
print(c.shape)

b = torch.rand(4, 64, 32)
b = b.unsqueeze(dim=1)
c = torch.matmul(a, b)
print(c.shape)

#power **2
a = torch.full([2, 2], 3)
print(a)
print(a.pow(2))
print(a**2)

# sqrt 开方 **(0.5)
aa = a.pow(2)
print(aa)
print(aa.sqrt())
print(aa**(0.5))

# 平方根的倒数
print(aa.rsqrt())

# exp log
a = torch.exp(torch.ones(2, 2))
print(a)

# log默认以e为底数
b = torch.log(a)
print(b)

print(torch.log10(a))

'''
Approximation
    .floor()向下取整 .ceil()向上取整
    .round()四舍五入
    .trunc()裁剪整数部分 .frac()裁剪小数部分
'''
a = torch.tensor(3.14)
print(a.floor())
print(a.ceil())
print(a.trunc())
print(a.frac())
print(a.round())

a = torch.tensor(3.5)
print(a.round())

''''
clamp
    gradient clipping
    (min)
    (min,max)
'''
grad = torch.rand(2, 3) * 20
print(grad)

# max 最大值
print(grad.max())

# median 中间值
print(grad.median())

print(grad.clamp(min=10, max=15))
