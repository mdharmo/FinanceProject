# Author: Mark Harmon
# Purpose: Let's just look at the spread of one data.  Both of level 1 and level 2 of the order book....
# This code will eventually
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

address1 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook07.pkl'
data1 = pkl.load(open(address1,'rb'))
address2 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook08.pkl'
data2 = pkl.load(open(address2,'rb'))
address3 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook09.pkl'
data3 = pkl.load(open(address3,'rb'))
address4 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook010.pkl'
data4 = pkl.load(open(address4,'rb'))
address5 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook011.pkl'
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

        if i ==0:
            spread1 = [spreadtemp1]
            spread2 = [spreadtemp2]
            askspread = [askspreadtemp]
            bidspread = [bidspreadtemp]
        else:
            spread1 += [spreadtemp1]
            spread2 += [spreadtemp2]
            askspread += [askspreadtemp]
            bidspread += [bidspreadtemp]


# Now we need to make the actual data for input into the model.
# Binary vector for which stock is being currently updated

totalnum = len(np.unique(data[:,-1])) + 10
stockidentifier = np.zeros(len(np.unique(data[:,-1])))
features = np.zeros(6)
featdat = np.zeros(len(data),totalnum)

# Here I make my data
for i in range(len(data)):

    si = data[i,-1]
    place = np.where(si==mystocks)[0]
    stockidentifier[place] = 1
    features[0] = data[i,2]
    features[1] = data[i,3]
    features[2] = data[i,5] - data[i,4]
    features[3] = data[i,6]
    features[4] = data[i,7]
    features[5] = data[i,8]
    features[6] = data[i,9]
    features[7] = data[i,11] - data[i,10]
    features[8] = data[i,12]
    features[9] = data[i,13]

    featdat[i,:] = np.concatenate((stockidentifier,features)).astype('float32')

# I'm going to create my actualy data as well...
# will have 13 times the number of stocks in data for each row (I will do actual time step
# since that makes actual since in context of the problem).  No need to predict
# something that has actually happened

n=13 # Thi is the number of features that I have for now
times = np.unique(data[:,1])
mystocks = np.unique(data[:,-1])
totalstocks = len(mystocks)
numfeat = totalstocks*n
featurevector = np.zeros(numfeat)-1
# going to make a row of data for each time step...

for i in range(len(times)):
    tempvec = np.where(data[:,0]==times[i])[0]
    tempdata = data[tempvec,:]

    tempstocks = np.unique(tempdata[:,-1])

    # Until I think of something better or faster, it seems that this
    # is the way I'm going to have to do it :/
    # I'm going to have to check as well if a stock shows up multiple times in a single time step
    # I have a feeling that this data just sucks
    count = 0
    for j in range(len(mystocks)):

        # This will tell us which part of the feature vector that we need to update
        if mystocks[j] == tempstocks[count]:
            # In this case update with information available
            count+=1

        else:
            # I'm goin
            pass




savename1 = '/home/mharmon/FinanceProject/ModelResults/eda/spread1/'
savename2 = '/home/mharmon/FinanceProject/ModelResults/eda/spread2/'
savename3 = '/home/mharmon/FinanceProject/ModelResults/eda/askspread/'
savename4 = '/home/mharmon/FinanceProject/ModelResults/eda/bidspread/'


pngcount = 0

for i in range(len(stocklist)):

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


