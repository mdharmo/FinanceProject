# Author: Mark Harmon
# Purpose: To look at the range of time values in the data
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
alltime1 = []
alltime2 = []
pngcount = 0
for j in range(5):


    address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+7) + 'list.pkl'



    data = pkl.load(open(address,'rb'))

    # Get time data 1
    time1 = [elem[0] for elem in data]
    time2 = [elem[0] for elem in data]


    alltime1 = alltime1 + time1
    alltime2 = alltime2 + time2

    time1 = np.array(time1,'float')
    time2 = np.array(time2,'float')

    delta1 = np.zeros(len(time1)-1)
    delta2 = np.zeros(len(time2)-1)

    for i in range(len(delta1)):
        delta1[i]=time1[i+1]-time1[i]
        delta2[i]=time2[i+1]-time2[i]

    data = []

    plotname = '/home/mharmon/FinanceProject/ModelResults/eda/time1/ind'+str(j+7)+'.png'
    plt.figure(pngcount)
    plt.plot(time1)
    plt.savefig(plotname)
    plt.close()
    pngcount+=1

    plotname = '/home/mharmon/FinanceProject/ModelResults/eda/time2/ind'+str(j+7)+'.png'
    plt.figure(pngcount)
    plt.plot(time2)
    plt.savefig(plotname)
    plt.close()
    pngcount+=1

    plotname = '/home/mharmon/FinanceProject/ModelResults/eda/delta1/ind'+str(j+7)+'.png'
    plt.figure(pngcount)
    plt.plot(delta1)
    plt.savefig(plotname)
    plt.close()
    pngcount+=1

    plotname = '/home/mharmon/FinanceProject/ModelResults/eda/delta2/ind' + str(j + 7) + '.png'
    plt.figure(pngcount)
    plt.plot(delta2)
    plt.savefig(plotname)
    plt.close()
    pngcount+=1

alltime1 = np.array(alltime1,'float')
alltime2 = np.array(alltime2,'float')

delta1 = np.zeros(len(alltime1) - 1)
delta2 = np.zeros(len(alltime2) - 1)

for i in range(len(delta1)):
    delta1[i] = alltime1[i + 1] - alltime1[i]
    delta2[i] = alltime2[i + 1] - alltime2[i]

plotname = '/home/mharmon/FinanceProject/ModelResults/eda/time1/all.png'
plt.figure(pngcount)
plt.plot(alltime1)
plt.savefig(plotname)
plt.close()
pngcount += 1

plotname = '/home/mharmon/FinanceProject/ModelResults/eda/time2/all.png'
plt.figure(pngcount)
plt.plot(alltime2)
plt.savefig(plotname)
plt.close()
pngcount += 1

plotname = '/home/mharmon/FinanceProject/ModelResults/eda/delta1/all.png'
plt.figure(pngcount)
plt.plot(delta1)
plt.savefig(plotname)
plt.close()
pngcount += 1

plotname = '/home/mharmon/FinanceProject/ModelResults/eda/delta2/all.png'
plt.figure(pngcount)
plt.plot(delta2)
plt.savefig(plotname)
plt.close()
pngcount += 1