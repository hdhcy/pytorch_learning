'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/15 16:19
'''
import torch

'''
statistics
    norm
    mean sum
    prod
    max,min,argmin,argmax
    kthvalue,topk
'''

'''
norm: 范数
    v.s.normalize,e.g.batch_norm
    matrix norm v.s. vector
'''
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)

# 0范数 非零元素的个数
print(a.norm(0), b.norm(0), c.norm(0))

# 1范数 所有元素的绝对值之和
print(a.norm(1), b.norm(1), c.norm(1))

# 2范数 所有元素的平方和在开根号
print(a.norm(2), b.norm(2), c.norm(2))

# dim表示维度，在第0维度上进行1范数的计算 取哪个维度的范数，哪个维度就会消掉
print(b.norm(1, dim=0))

print(b.norm(2, dim=1))

print(c.norm(1, dim=0))

print(c.norm(2, dim=0))

'''
mean: 均值
sum: 和
max: 最大值
min: 最小值
prod: 累乘
argmax: 最大值所在的索引，是将原来的矩阵进行打平之后的索引
argmin: 最小值所在的索引，是将原来的矩阵进行打平之后的索引
'''
a = torch.arange(8).view(2, 4).float()
print(a)

print(a.sum(), a.mean(), a.max(), a.min(), a.prod())

print(a.argmax(), a.argmin())

# 返回对应维度的最大或者最小值的索引
a = torch.rand(4, 10)
print(a[0])

print(a.argmax())

# dim表示维度，以argmax为例，对传入的维度上的值进行找出max的索引值并进行返回，即传入哪个维度，返回的结果中哪个维度就会消失
print(a.argmax(dim=1))
print(a.argmax(dim=0))

# dim and keepdim keepdim是保留原来的维度
print(a.max(dim=1))
print(a.argmax(dim=1))

print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))

# topk or k-th 返回自定义的前几个数据 默认求最大的几个
print(a.topk(3, dim=1))

# 通过设置largest=False来求最小的几个值
print(a.topk(4, largest=False))

# kthvalue 返回第几小的值
print(a.kthvalue(8, dim=1))  # 返回维度为1中第8小的值
print(a.kthvalue(3))  # 没有给定dim会默认为最后一个dim
print(a.kthvalue(3, dim=1))

'''
compare
    >,>=,<,<=,!=,==
    torch.eq(a,b)
'''
print(a > 0.5)
print(a.gt(0.5))
print(a != 0)

a = torch.ones(2, 3)
b = torch.rand(2, 3)
# .eq返回的是shape相同的矩阵中每一个位置的元素都进行判断所返回的布尔矩阵
print(a.eq(b))
print(a.eq(a))

# .equal() 返回整体上两个矩阵是否相等true/false
print(a.equal(b))
print(a.equal(a))
