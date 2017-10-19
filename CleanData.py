# Author: Mark Harmon
# Purpose: Here is where clean the data so that I can make the actual features for my data

import numpy as np
import pickle as pkl
import matplotlib
from scipy import stats

# A little spring cleaning first...
count = 0
for j in range(5):
    address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+7) + 'list.pkl'
    data = pkl.load(open(address,'rb'))

    # First, we have to filter out the stocks that don't have enough
    data = np.array(data,'float')
    stocks = np.unique(data[:,-1])
    for i in range(len(stocks)):
        temp = np.where(stocks[i]==data[:,-1])[0]
        askprice = data[temp,5]

        pricediff = np.zeros(len(askprice)-1)
        for k in range(len(askprice)-1):
            pricediff[k] = askprice[k+1]-askprice[k]

        nochange = float(len(np.where(pricediff==0)[0]))
        print(nochange/len(askprice))
        # If not enough entries, just delete the data
        if len(temp)<10000:
            data = np.delete(data,temp,0)

        elif (nochange/len(askprice))>0.99:
            data = np.delete(data,temp,0)



    address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0'+str(j+7)+'clean.pkl'
    pkl.dump(data,open(address,'wb'))

    print(len(data))

    if j ==0:
        totaldata = data
    else:
        totaldata = np.vstack((totaldata,data))

# Spread
stocksleft = np.unique(totaldata[:,-1])
spreadstd = np.zeros((len(stocksleft),2))
askstd = np.zeros((len(stocksleft),2))
bidstd = np.zeros((len(stocksleft),2))

for j in range(len(stocksleft)):
    temp = np.where(stocksleft[j]==totaldata[:,-1])[0]
    tempspread = totaldata[temp,5]-totaldata[temp,4]
    spreadstd[j,0] = np.std(tempspread)
    spreadstd[j,1]=stocksleft[j]
    askstd[j,0]=np.std(totaldata[temp,5])
    askstd[j,1]=stocksleft[j]
    bidstd[j,0]=np.std(totaldata[temp,4])
    bidstd[j,1]=stocksleft[j]

del totaldata
address = '/home/mharmon/FinanceProject/Data/cmedata/cleanstocks.pkl'
pkl.dump([stocksleft,spreadstd,bidstd,askstd],open(address,'wb'))