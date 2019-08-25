'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/18 15:56
'''
from visdom import Visdom

viz=Visdom()

#创建一个坐标y,x
viz.line([0.],[0.],win='train_loss',opts=dict(title='train loss'))

#multi-traces
viz.line([[0.0,0.0]],[0.0],win='test',opts=dict(title='test logss&acc.',legend=['loss','acc.']))


