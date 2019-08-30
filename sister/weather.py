'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/30 18:05
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
#第一部分 数据获取
'''
爬虫
    1.目标网址 http://www.weather.com.cn/weather/101300101.shtml
    2.获取网页内容(源代码)
    3.数据解析(提取) re xpath bs
    4.数据的保存
'''

def get_data(url):
    resp=requests.get(url)
    #print(resp)#<Response [200]>表示网络连接正常
    #print(resp.text)#文本形式的网页源代码
    #print(resp.content.decode('utf-8'))#二进制网页源代码
    html=resp.content.decode('utf-8')
    soup=BeautifulSoup(html,'html.parser')#默认的 可以使用xml但是得安装
    #print(soup)
    #注意class_
    li_list=soup.find_all('li',class_='sky')
    datas,conditions,temps,winds=[],[],[],[]
    for data in li_list:
        sub_data=data.text.split()
        #print(sub_data)
        datas.append(sub_data[0])
        #conditions.append(''.join(sub_data[1:3]))
        conditions.append(sub_data[1])
        temps.append(sub_data[2])
        winds.append(sub_data[3])

    table=pd.DataFrame()
    table['日期']=datas
    table['天气状况']=conditions
    table['气温']=temps
    table['风力']=winds

    return table




#第二部分 数据可视化
url=r'http://www.weather.com.cn/weather/101300101.shtml'
data=get_data(url)
data.to_csv('weather_data.csv')