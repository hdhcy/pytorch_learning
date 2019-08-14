'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/13 18:04
'''

def res(age):
    for a in range(1,age):
        for b in range(1,2*age-a):
            c=2*age-a-b
            if a*b*c==2450:
                return age,a,b,c

for i in range(48,52):
    print(res(i))






