# Author: Mark Harmon
# Purpose: To make the features for the cme model

import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sys import argv

def makelabels(sigmas,stocks,data,currentbook,prevbookdata,labelrange,allowance,which):

    labellength = len(data)/float(labelrange)

    # I need to look at the spread before, and decide.
    label = np.zeros((int(labellength)-1,len(stocks),5)).astype('uint8')
    label[:,:,2]=1
    beg = 0
    end = labelrange
    for m in range(int(labellength)-1):


        currentdata = data[beg:end,:]
        nextdata = data[beg+labelrange:end+labelrange,:]
        nextstocks = np.unique(nextdata[:,-1])

        for c in range(len(nextstocks)):
            # We need to do a couple checks on whether the stock appears more than once
            # for current data
            temp1 = np.where(nextstocks[c]==nextdata[:,-1])[0]
            if len(temp1>1):
                temp1 = temp1[-1]

            # For labels for spreads
            if which ==0:
                nextspread = nextdata[temp1,5]-nextdata[temp1,4]
                # Check for the existence of the stock here
                temp2 = np.where(nextstocks[c]==currentdata[:,-1])[0]
                # If this is true, then it is empty
                if temp2.size==0:
                    currentspread = nextspread
                elif len(temp2>1):
                    temp2 = temp2[-1]
                    currentspread = currentdata[temp2,5] - currentdata[temp2,4]

                # Now we compare the previous spread and current spreads to arrive at a label...
                diff = nextspread - currentspread
                sigma = sigmas[np.where(nextstocks[c] == sigmas[:,1]),0]
                #newlabel = int(diff/(sigma/allowance))+2

            # For labels for bid prices
            elif which==1:
                nextspread = nextdata[temp1, 4]
                # Check for the existence of the stock here
                temp2 = np.where(nextstocks[c] == currentdata[:, -1])[0]
                # If this is true, then it is empty
                if temp2.size == 0:
                    currentspread = nextspread
                elif len(temp2 > 1):
                    temp2 = temp2[-1]
                    currentspread = currentdata[temp2, 4]

                # Now we compare the previous spread and current spreads to arrive at a label...
                diff = nextspread - currentspread
                sigma = sigmas[np.where(nextstocks[c] == sigmas[:, 1]), 0]

            # For labels for ask prices
            else:
                nextspread = nextdata[temp1, 5]
                # Check for the existence of the stock here
                temp2 = np.where(nextstocks[c] == currentdata[:, -1])[0]
                # If this is true, then it is empty
                if temp2.size == 0:
                    currentspread = nextspread
                elif len(temp2 > 1):
                    temp2 = temp2[-1]
                    currentspread = currentdata[temp2, 5]

                # Now we compare the previous spread and current spreads to arrive at a label...
                diff = nextspread - currentspread
                sigma = sigmas[np.where(nextstocks[c] == sigmas[:, 1]), 0]
                # newlabel = int(diff/(sigma/allowance))+2

            if diff>0:
                newlabel=3
                if diff>1.*sigma:
                    newlabel=4
            elif diff<0:
                newlabel=1
                if diff<-1.*sigma:
                    newlabel=0
            else:
                newlabel=2
            # Find out where to place the new label...

            '''
            if newlabel>4:
                newlabel=4
            if newlabel<0:
                newlabel=0
            '''
            labelplace = np.where(nextstocks[c]==stocks)
            label[m, labelplace, 2] = 0
            label[m,labelplace,newlabel]=1



        beg=end
        end+=labelrange


    # Also, delete the last little bit of data to have continuous spread and labels
    amountleft = len(data)-beg
    needed = labelrange-amountleft
    beg = beg-needed
    currentdata = data[beg:,:]
    return label,currentdata


def main(which):
    # A little spring cleaning first...
    stocksaddress = '/home/mharmon/FinanceProject/Data/cmedata/cleanstocks.pkl'
    stocks,spreads,bids,asks = pkl.load(open(stocksaddress,'rb'))



    # First, we just extract our features including
    labelrange = 10
    previousrow = np.zeros(len(stocks)*12)
    prevdata = []


    if which ==0:
        del bids
        del asks
        spreads = spreads
        place = 'cmespread'
    elif which==1:
        del spreads
        del asks
        spreads = bids
        place = 'cmebid'
    else:
        del bids
        del spreads
        spreads = asks
        place = 'cmeask'

    for j in range(5):
        address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+7) + 'clean.pkl'
        data = pkl.load(open(address,'rb'))

        todelete = np.mod(len(data),labelrange)
        data = np.delete(data,np.arange(len(data)-todelete,len(data),1),0)
        #finalfeatures = np.zeros((len(data), 38 * 12)).astype('float32')
        # We do the features one step at a time here.
        # One thing that I think may be important is to have rolling values.  This means if for all the stocks in which there is not a new update
        # the values are simply equal to the previous one
        #beg = 0 + labelrange
        beg = 0
        end = 300000 + labelrange
        allowance=1.
        for k in range(int(len(data)/end)):

            featurechunk = np.zeros((300000, len(stocks) * 12)).astype('float32')

            for i in range(len(featurechunk)):
                temp = np.where(data[i+beg,-1]==stocks)[0][0]
                first = temp*12
                last = first + 12
                featurechunk[i,:] = previousrow
                featurechunk[i,first:last]=data[i+beg,2:14]
                previousrow = featurechunk[i,:]


            feataddress = '/home/mharmon/FinanceProject/Data/' + place + '/cmebook'+str(j+7)
            featurechunk = np.array(featurechunk,'float32')

            datachunk = data[beg:end,:]
            finallabels, prevdata = makelabels(spreads, stocks, datachunk, j, prevdata, labelrange,allowance,which)

            chunkadd = feataddress + str(k)+'.pkl'
            pkl.dump([featurechunk,finallabels],open(chunkadd,'wb'))
            beg=end-labelrange
            end+= 300000


        # This code connects the last part of each day to the next day.  I will probably need something like this
        # for the new data as well...

        if j<4:
            partialadd = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+8) + 'clean.pkl'
            datapart = pkl.load(open(partialadd,'rb'))
            datapart = datapart[0:labelrange]
            datachunk = np.vstack((data[beg:,:],datapart))

            # Get final chunk of features...
            featurechunk = np.zeros((len(datachunk)-labelrange,len(stocks)*12)).astype('float32')
            for i in range(len(featurechunk)):
                temp = np.where(datachunk[i,-1]==stocks)[0][0]
                first = temp*12
                last = first + 12
                featurechunk[i,:] = previousrow
                featurechunk[i,first:last]=data[i,2:14]
                previousrow = featurechunk[i,:]

            finallabels, prevdata = makelabels(spreads, stocks, datachunk, j, prevdata, labelrange,allowance,which)
            chunkadd = feataddress + str(k+1) + '.pkl'
            pkl.dump([featurechunk, finallabels], open(chunkadd, 'wb'))
        else:
            datachunk = data[beg:,:]
            finallabels,prevdata = makelabels(spreads,stocks,datachunk,j,prevdata,labelrange,allowance,which)

            featurechunk = np.zeros((len(datachunk) - labelrange, len(stocks) * 12)).astype('float32')
            for i in range(len(featurechunk)):
                temp = np.where(data[i,-1]==stocks)[0][0]
                first = temp*12
                last = first + 12
                featurechunk[i,:] = previousrow
                featurechunk[i,first:last]=data[i,2:14]
                previousrow = featurechunk[i,:]

            chunkadd = feataddress + str(k + 1) + '.pkl'
            pkl.dump([featurechunk, finallabels], open(chunkadd, 'wb'))

        del finallabels
        del data
        del featurechunk
        del datachunk

if __name__=='__main__':

    which = argv

    main(which)
