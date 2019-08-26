'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/18 15:56
'''

#run sever damon
#命令行：python -m visdom.server
import numpy as np
from visdom import Visdom
import time

viz=Visdom()
#Y X
viz.line([0.],[0.],win='train',opts=dict(title='x**2'))

image = viz.image(np.random.rand(3,256,256),opts={'title':'image1','caption':'How random.'})
for i in range(10):
    viz.image(np.random.randn( 3, 256, 256),win = image)
    time.sleep(0.5)


