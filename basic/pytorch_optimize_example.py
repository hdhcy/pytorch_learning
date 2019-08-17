'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/17 13:07
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

'''
himmelblau function
    f(x,y)=(x**2+y-11)**2+(x+y**2-7)**2
'''


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


'''
x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
print('x,y range',x.shape,y.shape)

#形成一张网格
X,Y=np.meshgrid(x,y)
print('X,Y maps',X.shape,Y.shape)

Z=himmelblau([X,Y])

fig=plt.figure()

axes3d=Axes3D(fig)

axes3d.plot_surface(X,Y,Z)

axes3d.set_xlabel('X')
axes3d.set_ylabel('Y')
axes3d.set_zlabel('Z')
axes3d.view_init(60,-30)

plt.show()
'''

# gradient descent
x = torch.tensor([0., 0., ], requires_grad=True)
# 完成优化的过程x'=x-lr*x的梯度    y'=y-lr*y的梯度
optimizer = torch.optim.Adam([x], lr=1e-3)

for step in range(20000):
    pred = himmelblau(x)

    # 梯度信息清零
    optimizer.zero_grad()
    pred.backward()
    # 调用一次optimizer.step()会跟新一次上述的优化过程
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x={}, f(x)={}'.format(step, x.tolist(), pred.item()))
