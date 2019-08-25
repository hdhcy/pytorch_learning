'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/21 15:12
'''
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

x=np.arange(-5.0,5.0,0.1)
y1=step_function(x)
plt.plot(x,y1)

y2=sigmoid(x)
plt.plot(x,y2)

y3=ReLU(x)
plt.plot(x,y3)

plt.ylim(-0.1,1.1)
plt.show()