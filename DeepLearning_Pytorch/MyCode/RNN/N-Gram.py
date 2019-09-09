'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/7 15:55
'''
import torch
from torch import optim,nn

CONTEXT_SIZE=2#依据的单词数量
EMBEDDING_DIM=10#词向量的维度

#我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram=[((test_sentence[i],test_sentence[i+1]),test_sentence[i+2])
         for i in range(len(test_sentence)-2)]
# print(len(trigram))
# print(trigram[0])

#使用set将重复的元素去掉
vocb=set(test_sentence)

word_to_idx={word:i for i,word in enumerate(vocb)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}

Length=len(word_to_idx)
#print(word_to_idx)
# print(idx_to_word)

class n_gram(nn.Module):
    def __init__(self,vocab_size,context_size=CONTEXT_SIZE,n_dim=EMBEDDING_DIM):
        super(n_gram,self).__init__()

        self.embed=nn.Embedding(vocab_size,n_dim)
        self.calssify=nn.Sequential(
            nn.Linear(context_size*n_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,vocab_size)
        )
    def forward(self,x):
        #x:[context_size]
        voc_embed=self.embed(x)#[context_size,n_dim]
        voc_embed=voc_embed.view(1,-1)#将两个此项链拼在一起[1,context_size*n_dim]
        out=self.calssify(voc_embed)#[1,vocab_size]
        return out

net=n_gram(Length)

criterion=nn.CrossEntropyLoss()
optimimzer=optim.Adam(net.parameters(),lr=1e-2)

# for word, label in trigram:
#
#     word = torch.LongTensor([word_to_idx[i] for i in word])
#
#     label=torch.LongTensor([word_to_idx[label]])
#

for e in range(100):
    train_loss=0
    for word,label in trigram:
        word=torch.LongTensor([word_to_idx[i] for i in word])
        label=torch.LongTensor([word_to_idx[label]])

        out=net(word)
        loss=criterion(out,label)
        train_loss+=loss.item()

        optimimzer.zero_grad()
        loss.backward()
        optimimzer.step()

    if(e+1)%20==0:
        print('Epoch: {}, Loss: {:.6f}'.format(e+1,train_loss/Length))

net=net.eval()

word,label=trigram[19]
print('input: {}   output: {}\n'.format(word,label))

word=torch.LongTensor([word_to_idx[i] for i in word])
out=net(word)
pred_label_idx=out.max(1)[1].item()
predict_word=idx_to_word[pred_label_idx]
print('real word is {},predicted word is {}'.format(label,predict_word))


word,label=trigram[67]
print('input: {}   output: {}\n'.format(word,label))

word=torch.LongTensor([word_to_idx[i] for i in word])
out=net(word)
pred_label_idx=out.max(1)[1].item()
predict_word=idx_to_word[pred_label_idx]
print('real word is {},predicted word is {}'.format(label,predict_word))









