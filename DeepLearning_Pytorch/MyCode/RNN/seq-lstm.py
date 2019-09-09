'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/7 19:41
'''
import torch
from torch import nn,optim
from torch.autograd import Variable

#简单的训练集
training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(),
                  ["NN", "V", "DET", "NN"])]

#对单词和标签进行编码
word_to_idx={}
tag_to_idx={}
for context,tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()]=len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()]=len(tag_to_idx)

# print(word_to_idx)
# print(tag_to_idx)

#对字母进行编码
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx={}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]]=i

# print(char_to_idx)

#构建训练数据
def make_sequence(x,dic):
    idx=[dic[i.lower()] for i in x]
    idx=torch.LongTensor(idx)
    return idx

# print(make_sequence('apple',char_to_idx).shape)
# print(training_data[0][0])
# print(make_sequence(training_data[1][0],word_to_idx))
# print(make_sequence(training_data[0][1],tag_to_idx))

class char_lstm(nn.Module):
    def __init__(self,n_char,char_dim,char_hiddlen):
        super(char_lstm,self).__init__()

        self.char_embed=nn.Embedding(n_char,char_dim)
        self.lstm=nn.LSTM(char_dim,char_hiddlen)

    def forward(self,x):
        #x(char numbers in a word,1)(3,1)
        x=self.char_embed(x)#(char numbers in a word,1,char_dim)(3,1,10)
        out,_=self.lstm(x)#(seq,batch,hidden)(3,1,50)
        return out[-1]  #(batch,hidden)(1,50)


class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim=10, word_dim=100,
                 char_hidden=50, word_hidden=128, n_tag=len(tag_to_idx)):
        super(lstm_tagger, self).__init__()
        self.word_embed = nn.Embedding(n_word, word_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)

    def forward(self, x, word):
        char = []
        for w in word:  # 对于每个单词做字符的 lstm
            #w: the dog ate the apple
            char_list = make_sequence(w, char_to_idx)
            #charlist#([19,7,4]) shape=[3]

            char_list = char_list.unsqueeze(1)  # (n_char,1) 满足 lstm 输入条件
            #char shape=[3,1]

            char_infor = self.char_lstm(Variable(char_list))  # (batch, char_hidden)
            #print(char_infor.shape)
            #char_infor shape[1,50]

            char.append(char_infor)
        char = torch.stack(char, dim=0)  #在指定维度连接新的张量 (seq, batch, feature)
        #char shape[words number in a wordlist,1,50]

        #x shape [1,5]
        x = self.word_embed(x)  # (batch, seq, word_dim)(1,5,word_dim)
        #x shape [1,5,100]

        x = x.permute(1, 0, 2)  # 改变顺序(seq,batch,word_dim)(5,1,word_dim)
        #x shape[5,1,100]

        x = torch.cat((x, char), dim=2)  # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        #x shape [5,1,150] [word_numbers(seq),batch,word_dim + char_hidden]

        x, _ = self.word_lstm(x)
        #x shape [5,1,128] [word_numbers(seq),batch,word_hidden]

        s, b, h = x.shape
        x = x.view(-1, h)  # 重新 reshape 进行分类线性层
        #x shape[word_numbers(seq)*batch,word_hidden]
        out = self.classify(x)
        #out shape[word_numbers(seq)*batch,n_tag]
        return out

net = lstm_tagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 开始训练
for e in range(10000):
    train_loss = 0
    for word, tag in training_data:
        #word ['The', 'dog', 'ate', 'the', 'apple']
        #word ['Everybody', 'read', 'that', 'book']

        word_list = make_sequence(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        #word_list [[0, 1, 2, 0, 3]] [[4, 5, 6, 7]]
        #word_list shape[1,5] [1,4]

        #tag ['DET', 'NN', 'V', 'DET', 'NN']
        #tag ['NN', 'V', 'DET', 'NN']

        tag = make_sequence(tag, tag_to_idx)
        #tag [0, 1, 2, 0, 1] [1, 2, 0, 1]
        #tag shape[5] [4]

        #好像也可以不用Variable
        # word_list = Variable(word_list)
        # tag = Variable(tag)

        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 200 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))

net = net.eval()
test_sent = 'Everybody ate the apple'
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())

print(out)
print(tag_to_idx)