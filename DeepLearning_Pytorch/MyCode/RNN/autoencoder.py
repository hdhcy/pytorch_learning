'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/8 16:21
'''
import os
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


im_tfs=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #标准化
])

train_set=datasets.MNIST('../data',transform=im_tfs)
train_data=DataLoader(train_set,batch_size=128,shuffle=True)

#定义网络
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()

        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,12),
            nn.ReLU(inplace=True),
            nn.Linear(12,3),#输出的code是3维，便于可视化
        )

        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(inplace=True),
            nn.Linear(12,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,28*28),
            nn.Tanh(),
        )
    def forward(self,x):
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return encode,decode

net=autoencoder()
# x=torch.randn(1,28*28)#batch size是1
# encode,decode=net(x)
criterion=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=1e-3)

def to_img(x):
    x=0.5*(x+1.)
    x=x.clamp(0,1)
    x=x.view(x.shape[0],1,28,28)
    return x

# 开始训练自动编码器
for e in range(100):
    for batch_idx ,(im, _ )in enumerate(train_data):
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        # 前向传播
        _, output = net(im)

        loss = criterion(output, im)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx+1) %128==0:
            print('Batch_idx={} Epoch: {}, Loss: {:.4f}'.format(batch_idx+1,e+1,loss.item()))

    if (e + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.item()))
        pic = to_img(output.cpu().data)
        if not os.path.exists('./simple_autoencoder'):
            os.mkdir('./simple_autoencoder')
        save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))


# 可视化结果
view_data = Variable((train_set.train_data[:200].type(torch.FloatTensor).view(-1, 28*28) / 255. - 0.5) / 0.5)
encode, _ = net(view_data)    # 提取压缩的特征值
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
values = train_set.train_labels[:200].numpy()  # 标签值
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色
    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()

code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]])) # 给一个 code 是 (1.19, -3.36, 2.06)
decode = net.decoder(code)
decode_img = to_img(decode).squeeze()
decode_img = decode_img.data.numpy() * 255
plt.imshow(decode_img.astype('uint8'), cmap='gray') # 生成图片 3











