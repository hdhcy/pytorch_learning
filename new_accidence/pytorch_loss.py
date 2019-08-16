'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/16 15:42
'''
import torch
from torch.nn import functional as F

'''
Typical Loss
    Mean Squared Error (MSE 均方差)
    Cross Entropy Loss
        binary
        multi-class
        +softmax
        Leave it to Logistic Regression Part
        
Gradient API
    torch.autograd.grad(loss,[W1,W2,...]
        [w1 grad,w2 grad,...]
    loss.backward()
        w1.grad
        w2.grad
'''
#autograd.grad
x=torch.ones(1)
w=torch.full([1],2)
print(x,w)

#第一个参数为prod，第二参数为label
mse=F.mse_loss(torch.ones(1),x*w)
print(mse)

#对w信息进行更新，使用require_grad_()后面的_表示input操作，对w变量进行更新，说明w需要grad操作
w.requires_grad_()
#也可以在声明的时候标注w是需要grad信息的 w=torch.full([1],2,requires_grad=True)

#在重新进行mse的声明，pytorch是一个动态图
mse=F.mse_loss(torch.ones(1),x*w)

#grad 求导函数，第一个参数为y(prod)，第二个参数为[x1,x2,x3...](w1,w2,...)
print(torch.autograd.grad(mse,[w]))#对mse进行w的偏导


#backward()
x=torch.ones(1)
w=torch.full([1],2,requires_grad=True)
mse=F.mse_loss(torch.ones(1),x*w)

#在最后的loss节点山调用backward时，它会自动的从后往前传播，会完成这条路径上所有需要梯度的tensor的一个grad梯度的计算方法，计算出来的grad不会在返回出来
mse.backward()
print(w.grad,w.norm())




