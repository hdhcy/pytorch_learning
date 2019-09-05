'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/9/5 10:25
'''
import pandas as pd
import matplotlib.pyplot as plt

def loadData(fileName):
    inFile=open(fileName,'r')
    epoch=[]
    number=[]
    loss=[]
    acc=[]
    for line in inFile:
        trainingSet=line.split('|')
        epoch.append(trainingSet[0].split()[0])
        number.append(int(trainingSet[0].split()[1]))
        loss.append(float(trainingSet[1].split(':')[1]))
        acc.append(float(trainingSet[2].split(':')[1].split()[0].split('%')[0])/100.0)
    return epoch,number,loss,acc

def plotData(epoch,number,loss,acc):
    plt.figure()
    plt.plot(number,loss,'r')
    plt.plot(number,acc,'b')
    plt.show()

epoch,number,loss,acc=loadData('log.txt')
plotData(epoch,number,loss,acc)