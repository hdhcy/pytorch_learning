'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/12 16:33
'''
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

W1 = np.random.randn(D_in, H)
W2 = np.random.randn(H, D_out)

learing_rate = 1e-6
for t in range(500):
    # Forward pass
    h = x.dot(W1)  # N*H
    h_relu = np.maximum(h, 0)  # N*H
    y_pred = h_relu.dot(W2)  # N*D_out

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_W2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(W2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weights of W1 and W2
    W1 -= learing_rate * grad_w1
    W2 -= learing_rate * grad_W2
