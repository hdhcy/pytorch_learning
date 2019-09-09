'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/6 13:46
'''
import torch
from torch import nn,optim
from torch.autograd import Variable

basic_rnn=nn.RNN(input_size=20,hidden_size=50,num_layers=2)
'''
print(basic_rnn.weight_ih_l0)
print(basic_rnn.weight_ih_l1)

print(basic_rnn.weight_hh_l0)
print(basic_rnn.weight_hh_l1)

print(basic_rnn.bias_ih_l0)
print(basic_rnn.bias_hh_l0)
'''
lstm=nn.LSTM(input_size=20,hidden_size=50,num_layers=2)
#print(lstm.weight_ih_l0.shape)#[hidden_size*4,input_size]

gru=nn.GRU(input_size=20,hidden_size=50,num_layers=2)
#print(gru.weight_ih_l0.shape)#[hidden_size*3,input_size]

toy_input=torch.randn(100,32,20)#seq batch feature
h_0=torch.randn(2,32,50)#layer*direction batch hidden_size

toy_output,h_n=basic_rnn(toy_input,h_0)
# print(toy_output.size())#seq,batch,hillde_size
# print(h_n.size())#layer*direction,batch,hiddlen_size

lstm_out,(h_n,c_n)=lstm(toy_input)
# print(lstm_out.size())#seq,batch,hillde_size
# print(h_n.size())#layer*direction,batch,hiddlen_size
# print(c_n.size())#layer*direction,batch,hiddlen_size

gru_out,h_n=gru(toy_input)
# print(gru_out.size())
# print(h_n.size())

word_to_idx={'hello':0,'world':1}
embeds=nn.Embedding(2,5)#(单词数量,词嵌入的维度)

hello_idx=torch.LongTensor([word_to_idx['hello']])
hello_idx=Variable(hello_idx)#shape=[1]

hello_embed=embeds(hello_idx)#shape=[1,词嵌入的维度]
print(hello_embed)
print(hello_embed.shape)

x=torch.randn(1,2,3)
print(x[-1].shape)