# Author: Mark Harmon
# Purpose: To make the features for the cme model

import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def makelabels(sigmas,stocks,data,currentbook,prevbookdata,labelrange,allowance):

    labellength = len(data)/float(labelrange)

    # I need to look at the spread before, and decide.
    label = np.zeros((int(labellength),38,5)).astype('uint8')
    label[:,:,2]=1
    beg = 0
    end = labelrange
    for m in range(int(labellength)):

        currentdata = data[beg:end,:]
        currentstocks = np.unique(currentdata[:,-1])

        if m>0 or (m==0 and currentbook>0):

            # Use previous data in current day or from previous day
            if m>0:
                previousdata = data[beg-labelrange:end-labelrange,:]
            else:
                previousdata = prevbookdata
            for c in range(len(currentstocks)):
                # We need to do a couple checks on whether the stock appears more than once
                # for current data
                temp1 = np.where(currentstocks[c]==currentdata[:,-1])[0]
                if len(temp1>1):
                    temp1 = temp1[-1]

                currentspread = currentdata[temp1,5]-currentdata[temp1,4]
                # Check for the existence of the stock here
                temp2 = np.where(currentstocks[c]==previousdata[:,-1])[0]
                # If this is true, then it is empty
                if temp2.size==0:
                    previousspread = currentspread
                elif len(temp2>1):
                    temp2 = temp2[-1]
                    previousspread = previousdata[temp2,5] - previousdata[temp2,4]

                # Now we compare the previous spread and current spreads to arrive at a label...
                diff = previousspread - currentspread
                sigma = sigmas[np.where(currentstocks[c] == sigmas[:,1]),0]
                #newlabel = int(diff/(sigma/allowance))+2
                if diff>0:
                    newlabel=3
                    if diff>sigma:
                        newlabel=4
                elif diff<0:
                    newlabel=1
                    if diff<-sigma:
                        newlabel=0
                else:
                    newlabel=2
                # Find out where to place the new label...

                labelplace = np.where(currentstocks[c]==stocks)
                label[m, labelplace, 2] = 0
                label[m,labelplace,newlabel]=1
        else:
            label[m,:,2]=1



        beg=end
        end+=labelrange


    # Also, delete the last little bit of data to have continuous spread and labels
    amountleft = len(data)-beg
    needed = labelrange-amountleft
    beg = beg-needed
    currentdata = data[beg:,:]
    print(len(currentdata))
    return label,currentdata

# A little spring cleaning first...
stocksaddress = '/home/mharmon/FinanceProject/Data/cmedata/cleanstocks.pkl'
stocks,spreads = pkl.load(open(stocksaddress,'rb'))

# First, we just extract our features including
labelrange = 50
previousrow = np.zeros(38*12)
prevdata = []


for j in range(5):
    address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+7) + 'clean.pkl'
    data = pkl.load(open(address,'rb'))

    todelete = np.mod(len(data),labelrange)
    data = np.delete(data,np.arange(len(data)-todelete,len(data),1),0)
    #finalfeatures = np.zeros((len(data), 38 * 12)).astype('float32')
    # We do the features one step at a time here.
    # One thing that I think may be important is to have rolling values.  This means if for all the stocks in which there is not a new update
    # the values are simply equal to the previous one
    for i in range(len(data)):
        temp = np.where(data[i,-1]==stocks)[0][0]
        beg = temp*12
        end = beg + 12
        finalfeatures[i,:] = previousrow
        finalfeatures[i,beg:end]=data[i,2:14]
        previousrow = finalfeatures[i,:]


    feataddress = '/home/mharmon/FinanceProject/Data/cmenew/cmebook'+str(j+7)
    finalfeatures = np.array(finalfeatures,'float32')

    # Now we need to split it up into chunks to be able to save it...

    beg = 0 + labelrange
    end = 500000 + labelrange
    allowance=1.
    for k in range(int(len(data)/end)):
        datachunk = data[beg:end,:]
        featurechunk = finalfeatures[beg-labelrange:end-labelrange,:]
        finallabels, prevdata = makelabels(spreads, stocks, datachunk, j, prevdata, labelrange,allowance)

        chunkadd = feataddress + str(k)+'.pkl'
        pkl.dump([featurechunk,finallabels],open(chunkadd,'wb'))
        beg=end
        end+= 500000

    # This code connects the last part of each day to the next day.  I will probably need something like this
    # for the new data as well...
    if j<4:
        partialadd = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+8) + 'clean.pkl'
        datapart = pkl.load(open(partialadd,'rb'))
        datapart = datapart[0:labelrange]
        datachunk = np.vstack((data[beg:,:],datapart))
        finallabels, prevdata = makelabels(spreads, stocks, datachunk, j, prevdata, labelrange,allowance)
        featurechunk = finalfeatures[beg:,:]
        chunkadd = feataddress + str(k+1) + '.pkl'
        pkl.dump([featurechunk, finallabels], open(chunkadd, 'wb'))
    else:
        datachunk = data[beg:,:]
        finallabels,prevdata = makelabels(spreads,stocks,datachunk,j,prevdata,labelrange,allowance)
        featurechunk = finalfeatures[beg:-labelrange,:]
        chunkadd = feataddress + str(k + 1) + '.pkl'
        pkl.dump([featurechunk, finallabels], open(chunkadd, 'wb'))

    del finalfeatures
    del finallabels
    del data
    del featurechunk
    del datachunk
    del datapart



