# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# So, let's make the actual training data as well.

import numpy as np
import pickle as pkl
import sys

def label_make(data,diff,sigma,seqsize):
    # 5 stocks and 5 labels...
    labels = np.zeros((len(data)-seqsize-1,5,5))

    beg = seqsize-1
    totaldata = []
    for i in range(len(labels)):

        testtemp = diff[beg+i,:]
        totaldata += [data[i:beg+i+1,:].tolist()]
        for j in range(5):

            sigtemp = sigma[j]


            if testtemp[j]<-sigtemp:
                labels[i,j,0]=1
            elif testtemp[j]<0.:
                labels[i,j,1]=1
            elif testtemp[j]==0:
                labels[i,j,2]=1
            elif testtemp[j] <=sigtemp:
                labels[i,j,3]=1
            else:
                labels[i,j,4]=1

    return labels,totaldata

def main(seqsize):
    address = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'
    data,dates = pkl.load(open(address,'rb'))
    diff = np.zeros((len(data)-1,5))
    # I forgot to normalize the data... Pretty dumb of me
    for i in range(5):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])

    # Calculate differences
    for i in range(len(data)-1):
        diff[i,:] = data[i+1,:]-data[i,:]

    # Calculate my sigma values
    sigma = np.zeros(5)
    for i in range(5):
        sigma[i] = np.std(diff[:,i])

    # Now make the actual labels
    sigsave = '/home/mharmon/FinanceProject/Data/tickdata/sigma.pkl'
    pkl.dump(sigma,open(sigsave,'wb'))

    length = int(len(data)/2.)
    labels,totaldata = label_make(data[:length],diff[:length],sigma,seqsize)
    finaldates = dates[seqsize+1:]
    totaldata = np.array(totaldata,'float32')
    labels = np.array(labels,'uint8')
    pkl.dump([totaldata,labels,finaldates],open('/home/mharmon/FinanceProject/Data/tickdata/train' + str(seqsize)+'.pkl','wb'))


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    main(seqsize)