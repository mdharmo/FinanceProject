# Author: Mark Harmon
# Purpose: To make the features for the cme model

import numpy as np
import pickle as pkl
import matplotlib

# A little spring cleaning first...
sigma = np.zeros(38)
stocksaddress = '/home/mharmon/FinanceProject/Data/cmedata/cleanstocks.pkl'
stocks,spreads = pkl.load(open(stocksaddress,'rb'))
Row = np.zeros((38, 12)) - 1
OneLabel = np.zeros((38, 5))
OneLabel[:,2]=1
exists = 0
for j in range(5):
    address = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j+7) + 'clean.pkl'
    data = pkl.load(open(address,'rb'))
    # Grab my times and convert to actual seconds
    times = data[:,0]/1000000000

    # Get total difference, that's the number of seconds we need
    # FYI, int is a floor function, which is what I prefer anyways
    delta = times[-1]-times[0]
    begtime = times[0]

    # This doesn't necessarily cover all the changes.  There will be some that happen within a second
    # I'll have to take care of this amount of time after the for loop
    finalfeatures = np.zeros((int(delta),38,12))
    finallabels = np.zeros((int(delta),38,5))
    for i in range(int(delta)):
        # Get one second of data first
        endtime = begtime +1
        begplace = np.where(times>=begtime)[0][0]
        endplace = np.where(times>=endtime)[0][0]
        tempdata = data[begplace:endplace,:]

        # Here fill in what I need to for features
        # First, get the list of stocks in this temp data:

        tempstocks = np.unique(tempdata[:,-1])

        for k in range(len(tempstocks)):

            lastspot = np.where(tempstocks[k]==tempdata[:,-1])[0]
            # Checking to make sure that we get the last time spot...
            if len(lastspot)!=0:
                lastspot = lastspot[-1]

            # Find which spot to put the feature.  I could probably get rid of a couple of these loops,
            # But I think it's better this way for now for readability and easy diagnoses for bugs
            featurespot = np.where(tempstocks[k]==spreads[:,1])[0][0]
            Row[featurespot,:] = tempdata[lastspot,2:14] # Nothing fancy for now, but this is where I can start making changes

            # Need to make a few checks for creating the label.
            # 1). I need to compare to previous spread, so much sure that exists
            # 2). Make sure previous spread is not one of the -1 values

            findprevious = np.where(tempstocks[k]==data[:,-1])
            if i==0:
                previousrow = np.zeros((38,12))-1
            else:
                previousrow = finalfeatures[i-1,:,:]
            # Check for the spread value
            if previousrow[featurespot,5]!=-1:
                previousspread = previousrow[featurespot, 5] - previousrow[featurespot, 4]
                currentspread = Row[featurespot,5]-Row[featurespot,4]

                # This is organzied in such a way that I don't have to have to test for each if statement
                OneLabel[featurespot,:]=np.zeros(5)
                if currentspread-previousspread<-2*spreads[featurespot,0]:
                    OneLabel[featurespot,0]=1
                elif currentspread-previousspread<-spreads[featurespot,0]:
                    OneLabel[featurespot,1]=1
                elif currentspread-previousspread<spreads[featurespot,0]:
                    OneLabel[featurespot,2]=1
                elif currentspread-previousspread>=spreads[featurespot,0]:
                    OneLabel[featurespot,3]=1
                elif currentspread-previousspread>=2*spreads[featurespot,0]:
                    OneLabel[featurespot,5]=1


        finalfeatures[i,:,:]=Row
        finallabels[i,:,:]=OneLabel
        begtime = endtime

    if j ==0:
        totalfeatures = finalfeatures
        totallabels = finallabels
    else:
        totalfeatures = np.vstack((totalfeatures,finalfeatures))
        totallabels = np.vstack((totallabels,finallabels))



homeaddress = '/home/mharmon/FinanceProject/Data/cmefinal/totalfinaldata.pkl'
pkl.dump([totalfeatures,totallabels],open(homeaddress,'wb'))

