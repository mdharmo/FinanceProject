# Author: Mark Harmon
# Purpose: Let's just look at the spread of one data.  Both of level 1 and level 2 of the order book....
# This code will eventually
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

address1 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook07clean.pkl'
data1 = pkl.load(open(address1,'rb'))
address2 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook08clean.pkl'
data2 = pkl.load(open(address2,'rb'))
address3 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook09clean.pkl'
data3 = pkl.load(open(address3,'rb'))
address4 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook010clean.pkl'
data4 = pkl.load(open(address4,'rb'))
address5 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook011clean.pkl'
data5 = pkl.load(open(address5,'rb'))

data = np.vstack((data1,data2,data3,data4,data5))
del data1,data2,data3,data4,data5

stocklist = np.unique(data[:,-1])
badlist = []
for i in range(len(stocklist)):

    temp = np.where(data[:,-1]==stocklist[i])[0]

    mymed = np.median(temp)
    a = np.where(temp!=mymed)[0]

    # Check if the stock has enough records
    if len(temp)<10000:
        badlist +=[i]

        data = np.delete(data,temp,0)

    # Check to make sure that the stock is at least a little volatile
    elif len(a) <= int(len(temp)/4.):
        badlist+=[i]
        data = np.delete(data,temp,0)

    else:
        spreadtemp1 = data[temp,5] - data[temp,4]
        spreadtemp2 = data[temp,11] - data[temp,10]
        askspreadtemp = data[temp,11] - data[temp,5]
        bidspreadtemp = data[temp,4] - data[temp,10]
        askprice1temp = data[temp,5]
        bidprice1temp = data[temp,4]
        if i ==0:
            spread1 = [spreadtemp1]
            spread2 = [spreadtemp2]
            askspread = [askspreadtemp]
            bidspread = [bidspreadtemp]
            askprice1 = [askprice1temp]
            bidprice1 = [bidprice1temp]
        else:
            spread1 += [spreadtemp1]
            spread2 += [spreadtemp2]
            askspread += [askspreadtemp]
            bidspread += [bidspreadtemp]
            askprice1 += [askprice1temp]
            bidprice1 += [bidprice1temp]


spread1 = np.array(spread1)
spread2 = np.array(spread2)
askspread = np.array(askspread)
bidspread = np.array(bidspread)


savename1 = '/home/mharmon/FinanceProject/ModelResults/eda/spread1/'
savename2 = '/home/mharmon/FinanceProject/ModelResults/eda/spread2/'
savename3 = '/home/mharmon/FinanceProject/ModelResults/eda/askspread/'
savename4 = '/home/mharmon/FinanceProject/ModelResults/eda/bidspread/'
savename5 = '/home/mharmon/FinanceProject/ModelResults/eda/hist/'
savename6 = '/home/mharmon/FinanceProject/ModelResults/eda/askprice/'
savename7 = '/home/mharmon/FinanceProject/ModelResults/eda/bidprice/'
pngcount = 0

for i in range(len(stocklist)):

    mycount = np.where(np.median(spread1[i]) == spread1[i])[0]
    print(100*len(mycount)/float(len(spread1[i])))

    name1 = 'Stock ' + str(i) + ' Level 1 Ask Price'
    figsave = savename6 + 'Stock' +str(i) + 'Ask1.png'
    plt.figure(pngcount)
    plt.plot(askprice1[i])
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Level 1 Bid Price'
    figsave = savename7 + 'Stock' +str(i) + 'Bid1.png'
    plt.figure(pngcount)
    plt.plot(bidprice1[i])
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Level 1 Spread Hist'
    figsave = savename5 + 'Stock' +str(i) + 'Spread1Hist.png'
    plt.figure(pngcount)
    plt.hist(spread1[i],bins=10)
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Level 1 Spread'
    figsave = savename1 + 'Stock' +str(i) + 'Spread1.png'
    plt.figure(pngcount)
    plt.plot(spread1[i])
    plt.xlabel('Update')
    plt.ylabel('Spread Value')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Level 2 Spread'
    figsave = savename2 + 'Stock' +str(i) + 'Spread2.png'
    plt.figure(pngcount)
    plt.plot(spread2[i])
    plt.xlabel('Update')
    plt.ylabel('Spread Value')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Ask Spread'
    figsave = savename3 + 'Stock' +str(i) + 'AskSpread.png'
    plt.figure(pngcount)
    plt.plot(askspread[i])
    plt.xlabel('Update')
    plt.ylabel('Spread Value')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

    name1 = 'Stock ' + str(i) + ' Bid Spread'
    figsave = savename4 + 'Stock' +str(i) + 'BidSpread.png'
    plt.figure(pngcount)
    plt.plot(bidspread[i])
    plt.xlabel('Update')
    plt.ylabel('Spread Value')
    plt.title(name1)
    plt.savefig(figsave)
    plt.close()
    pngcount+=1

