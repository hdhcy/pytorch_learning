'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/28 8:37
'''
import numpy as np
import pandas as pd

'''
locals:表示问路径，传入的是最好要使用以r开始的原始路径 或者加入转译字符\  C:\\Users\\98276\\Downloads\\sister.xls
year:表示学年例如2016-2017
term:表示学期例如2表示第二学期
'''


class Gpa:
    def __init__(self, locals, year=None, term=None):
        self.locals = locals
        self.year = year
        self.term = term

    # 计算全部学期的绩点
    def all_compute(self):
        data = pd.read_excel(self.locals)
        xuefen = data['学分']
        jidian = data['绩点']
        print('总学分: ', xuefen.sum(),'课程数: ',xuefen.size)
        #print('总学分*总绩点',(xuefen * jidian).sum())
        print('总绩点: ', (xuefen * jidian).sum() / xuefen.sum())

    # 计算指定学年的绩点
    def year_compute(self):
        data = pd.read_excel(self.locals)
        table = data.loc[:, ['学年', '学分', '绩点']]
        year_table = table[table.学年 == self.year]
        xuefen = year_table['学分']
        jidian = year_table['绩点']
        print(self.year, '年的学分: ', xuefen.sum())
        print(self.year, '年的绩点: ', (xuefen * jidian).sum() / xuefen.sum())

    # 计算指定学期指定学年的绩点
    def confim_compute(self):
        data = pd.read_excel(self.locals)
        table = data.loc[:, ['学年', '学期', '学分', '绩点']]
        com_table = table[table.学年 == self.year]
        com_table = com_table[com_table.学期 == self.term]
        xuefen = com_table['学分']
        jidian = com_table['绩点']
        print(self.year, '年 第', self.term, '学期的学分: ', xuefen.sum())
        print(self.year, '年 第', self.term, '学期的绩点: ',
              (xuefen * jidian).sum() / xuefen.sum())
