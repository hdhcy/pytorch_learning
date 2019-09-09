'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/27 21:27
'''
import xlrd
import numpy as np
from Gpa import Gpa
import pandas as pd

test=pd.read_excel(r'C:\Users\98276\Downloads\sister.xls')
table=test.loc[:,['学年','学期','学分','绩点']]
table1=table[table.学年=='2016-2017']
table2=table1[table1.学期==2]
#print(table2)


data=xlrd.open_workbook('C:\\Users\\98276\\Downloads\\sister.xls')
table=data.sheet_by_index(0)

xuefen_list=table.col_values(5)
chengji_list=table.col_values(8)
del xuefen_list[0]
del chengji_list[0]

n_xuefen_list=[]
for i in xuefen_list:
    n_xuefen_list.append(float(i))

n_chengji_list=[]
for i in chengji_list:
    n_chengji_list.append(float(i))

xuefen=np.array(n_xuefen_list)
chengji=np.array(n_chengji_list)

xuefen=np.mat(xuefen)
chengji=np.mat(chengji)

xuefen_sum=xuefen.sum()
#print('总学分:  ',xuefen_sum)
total=xuefen*chengji.T

#print('绩点:  ',(total/xuefen_sum))

gpa=Gpa(r'C:\Users\98276\Downloads\chengji.xls')
gpa.all_compute()



