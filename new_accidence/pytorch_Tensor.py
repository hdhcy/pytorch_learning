'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/13 20:42
'''
import numpy as np
import torch

# 大写Tensor给shape 小写tnesor给数据

# import from numpy
a = np.array([2, 3.3])
b = torch.from_numpy(a)
print(b)

a = np.ones([2, 3])
b = torch.from_numpy(a)
print(b)

# import from List
a = torch.tensor([2., 3.2])
print(a)

b = torch.FloatTensor(2, 3)
c = torch.FloatTensor([2., 3.2])
print(b)
print(c)

d = torch.tensor([[2., 3.2],
                  [1., 22.3]])
print(d)

# uninitialized
a = torch.empty(1)
print(a)

b = torch.Tensor(2, 3)
print(b)

c = torch.IntTensor(2, 3)
print(c)

d = torch.FloatTensor(2, 3)
print(d)

# set default type
a = torch.tensor([1.2, 3]).type()
print(a)

torch.set_default_tensor_type(torch.DoubleTensor)
b = torch.tensor([1., 3]).type()
print(b)

# rand/rand_like,randint

# rand 从0和1之间随机的均匀分布出来
a = torch.rand(3, 3)
print(a)

# rand_like 把传入的参数的shape读取出来然后送给新的变量
b = torch.rand_like(a)
print(b)

# randint 可以随机均匀产生自己自定义的范围内的数据[min,max)
# min,max,shape
c = torch.randint(1, 10, [2, 3])
print(c)

# randint_like 传入的参数,min,max
d = torch.randint_like(c, 2, 6)
print(d)

# randn
# N(0,1) 正态分布 得到均值为0，方差为1的数据
a = torch.randn(3, 3)
print(a)

# normal 自定义范围，均值和方差
b = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print(b)

c = b.reshape(2, 5)
print(c)

# full 把全部的tensor设置为一个元素
a = torch.full([2, 3], 7)
print(a)

# 一个常量
b = torch.full([], 8)
print(b)

# 一个dim为1的变量
c = torch.full([2], 9)
print(c)

# arange/range   生成等差数列
a = torch.arange(0, 10)
print(a)

b = torch.arange(0, 10, 2)
print(b)

# range已经被弃用
#c=torch.range(1, 4, 0.5)
# print(c)

# linespace/logspace 等分 注意为左闭右闭[min,max]
a = torch.linspace(0, 10, steps=4)
print(a)

b = torch.linspace(0, 10, steps=10)
print(b)

c = torch.linspace(0, 10, steps=11)
print(c)

# logspace 表示从10的min次方到10的max次方,左闭右闭
d = torch.logspace(0, 10, steps=10)
print(d)

e = torch.logspace(0, -1, steps=10)
print(e)

# ones/zeros/eye
a = torch.ones(3, 3)
print(a)

b = torch.zeros(3, 3)
print(b)

# 不是对角矩阵按小的来填充1，eye只适合二维矩阵
c = torch.eye(3, 2)
print(c)

# 也可以传入一个数字表示对角矩阵
d = torch.eye(3)
print(d)

# 注意like中传入的参数并不要求和要like的数据相同，只是要它的shape
e = torch.ones_like(a)
print(e)

f = torch.zeros_like(a)
print(f)

# randperm随机打散[0,max)
a = torch.randperm(10)
print(a)

a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print(idx)
print(a)

# indexing
a = torch.rand(4, 3, 28, 28)
print(a)
print(a.shape)
print(a[0].shape)
print(a[0][0].shape)
print(a[0][0][0].shape)
# 表示一个标量
print(a[0][0][0][0].shape)
print(a[0][0][0][0])

# select first/last N
a = torch.rand(4, 3, 28, 28)
print(a.shape)

# 第一个维度[0,2)
print(a[:2].shape)

# 第一个维度[0,2) 第二个维度[0,1)
print(a[:2, :1, :, :].shape)

# 第一个维度[0,2) 第二个维度[1,max)
print(a[:2, 1:].shape)

# 第二个维度的-1表示从最后一个元素取值
print(a[:2, :-1].shape)

# select by steps    negative step not yet supported
'''
1.  :   all
2.  m:n  [m,n) :n  [0,n) n: [n,max)
3.  m:n:stpe [m,n)每隔一个step取一个数    ::step (0,max]每隔一个step取一个数    step不写的话默认为1
'''
print(a[:, :, 0:28:2, 0:28:2].shape)

print(a[:, :, ::2, ::2].shape)

# select by specific index
print(a.shape)
'''
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices)
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> torch.index_select(x, 1, indices)
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])
'''
print(a.index_select(0, torch.tensor([0, 2])).shape)

print(a.index_select(1, torch.tensor([1, 2])).shape)

print(a.index_select(2, torch.arange(24)).shape)

# ... 表示任意多的维度，根据情况来说
print(a.shape)

# 表示取所有维度
print(a[...].shape)

# 表示第二三四维度
print(a[0, ...].shape)

print(a[:, 1, ...].shape)

print(a[0, ..., ::2].shape)

print(a[..., :2].shape)

# select by mask mask表示掩码
x = torch.randn(3, 4)
print(x)

# ge表示选取大于0.5的数据，返回整个坐标，大于0.5用于True表示，小于0.5用False表示
mask = x.ge(0.5)
print(mask)
print(mask.dtype)

y = torch.masked_select(x, mask)
print(y)
print(y.shape)

# select by flatten index
src = torch.tensor([[4, 3, 5],
                    [6, 7, 8]])
# take会先将数据打平成一行数据,索引从0开始
a = torch.take(src, torch.tensor([0, 2, 5]))
print(a.shape)

# view/reshpe    view和reshpe变换要保证其数据个数相同，满足数学变换
a = torch.rand(4, 1, 28, 28)
print(a.shape)

b = a.view(4, 28 * 28)
print(b)
print(b.shape)

print(a.view(4 * 28, 28).shape)

print(b.view(4 * 1, 28, 28).shape)

b = a.view(4, 784)
c = b.view(4, 28, 28, 1)  # logic bug 容易造成维度信息的缺失

# flexible but prone to current
# a.view(4,783)

# squeeze v.s. unsequeeze
# 减少和增加维度

# unsequence
a = torch.rand(4, 1, 28, 28)
print(a.shape)

# unsqueeze接受参数为position/index,表示在哪个位置(维度)上进行插入,注意插入后的整个index总长度会加一
print(a.unsqueeze(0).shape)

# 在最后一个位置插入
'''
索引的位置 建议不要使用负数
 0, 1, 2, 3, 4
-5,-4,-3,-2,-1
'''
print(a.unsqueeze(-1).shape)

a = torch.tensor([1.2, 2.3])
print(a.shape)

b = a.unsqueeze(-1)
print(b)
print(b.shape)

c = a.unsqueeze(0)
print(c)
print(c.shape)

b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)

# [32]=>[32,1]=>[32,1,1]=>[1,32,1,1]
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

# squeeze 接受index/position 如果什么都不传入，能删减多少删减多少
print(b.shape)

# 能挤压的都挤压 dim为1的
print(b.squeeze().shape)

print(b.squeeze(0).shape)

print(b.squeeze(-1).shape)

# 该维度不是1，不能被挤压
print(b.squeeze(1).shape)

print(b.squeeze(-4).shape)

# expand/repeat 维度的扩展
# expand只是改变了理解方式不增加新的数据，repeat会增加新的数据，把所有的数据都有cpoy一遍

# expand
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)

'''
[1,32, 1, 1]
[4,32,14,14]
只能由1转换成M    例如1->M
不能用不是1的数进行转换 例如3->M
'''
c = b.expand([4, 32, 14, 14])
print(c.shape)

# 维度不变用-1表示
d = b.expand(-1, 32, -1, -1)
print(d.shape)

# bug已经修复，不能用除-1以外的负数
# e=b.expand(-1,32,-1,-4)
# print(e)

# repeat 接受的参数为每一个维度上面的数据要copy的次数
b = torch.rand(1, 32, 1, 1)
print(b.shape)

c = b.repeat(4, 32, 1, 1)
print(c.shape)

d = b.repeat(4, 1, 1, 1)
print(d.shape)

e = b.repeat(4, 1, 32, 32)
print(e.shape)

# .t 进行矩阵的转置，只能适用于2D的矩阵
a = torch.randn(3, 4)
print(a)
print(a.t())

#transpose 接受维度索引进行交换
a=torch.randn(4,3,32,32)
print(a.shape)

#tranpose涉及到数据的交换，数据被打乱变成不连续的，可以使用.contiguous()函数把它重新的变成连续
#view操作会丢失维度信息，造成数据污染
a1=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)

#需要进行重新一次的transpose
a2=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)

print(a1.shape,a2.shape)

#使用eq()函数来比较数据内容是不是一致，用all()函数来确保所有数据都一致才返回True
print(torch.all(torch.eq(a,a1)))

print(torch.all(torch.eq(a,a2)))

#permute
a=torch.rand(4,3,28,28)
print(a.transpose(1,3).shape)

b=torch.rand(4,3,28,32)
print(b.transpose(1,3).shape)

c=b.transpose(1,3).transpose(1,2)
print(c.shape)

#permute操作比较简单，传入所有index按照自己的需求 保证所有的index都出现且只出现一次
d=b.permute(0,2,3,1)
print(d.shape)

#broatcasting
'''
how to realize this? A:[4,32,8]+[5]
[1].unsqueeze(0).unsqueeze(0).expand_as(A)

why boratcasting
for actual demanding
memory consumption

is it broadcasting-able?
Match from Last dim!
    if current dim=1,expand to same
    if either has no dim,insert one dim and expand to same
    other Not broadcasting-able
    
How to understand this behavior?
    When it has no dim
        treat it as all own the same
        [class,student,scores]+[scores]
    When it has dim of size 1
        treat it shared by all
        [class,student,scores]+[student,1]
'''
#merge or split 合并与切割
#cat cat要求除了要合并的维度之外，其他维度必须相同
#[class1-4,students,scores]
a=torch.rand(4,32,8)
#[class5-9,students,scores]
b=torch.rand(5,32,8)

#第一个参数为list包含了所有要合并的tensor,第二个参数dim为要合并的维度
c=torch.cat([a,b],dim=0)
print(c.shape)

a1=torch.rand(4,3,32,32)
a2=torch.rand(5,3,32,32)

b=torch.cat([a1,a2],dim=0)
print(b.shape)

a2=torch.rand(4,1,32,32)
#b=torch.cat([a1,a2],dim=0)
#print(b.shape)

a1=torch.rand(4,3,16,32)
a2=torch.rand(4,3,16,32)

b=torch.cat([a1,a2],dim=2)
print(b.shape)

