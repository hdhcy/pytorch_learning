'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/26 15:06
'''
import numpy as np
from tensorboardX import SummaryWriter

writer=SummaryWriter()
for epoch in range(100):
    writer.add_scalar('scalar/test',np.random.rand(),epoch)
    writer.add_scalars('scalar/scalar_test',{'xsinx':epoch*np.sin(epoch),'xcosx':epoch*np.cos(epoch)},epoch)


