# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# So, let's make the actual training data as well.

import numpy as np
import pickle as pkl
import sys

def label_make(data,diff,sigma,seqsize,step,window):
    # 5 stocks and 5 labels...
    # Binary version is a single label...
    stocks = 5
    lablenint = int((len(data)-seqsize-1)/step)
    lablenfloat = (len(data)-seqsize-1)/float(step)
    test_num = int((window-step-(lablenfloat-lablenint)*step)/step)
    lablen = lablenint - np.max((0,test_num))
    labels = np.zeros((lablen,window,5,5))

    # The only thing that really matters is the sigma values.
    # I need to calculate new sigma values every so often.
    # What I need is a counter for when I get over the limit...

    week_count = 1
    week_len = 2016
    beg = seqsize-1
    totaldata = []
    factor = 2
    for i in range(len(labels)):

        testtemp = diff[beg+i*step:beg+i*step+window,:]
        totaldata += [data[i*step:beg+i*step+1,:].tolist()]

        if beg+i*step+window>factor*week_len:
            week_count+=1
            factor+=1
        # Need to calculate a new sigma value after each week
        sigma = np.zeros(stocks)
        for m in range(stocks):
            sigma[m] = np.std(diff[(week_count-1)*week_len:week_count*week_len])

        for k in range(window):

            for j in range(5):

                sigtemp = sigma[j]

                if testtemp[k,j]<-sigtemp:
                    labels[i,k,j,0]=1
                elif testtemp[k,j]<0.:
                    labels[i,k,j,1]=1
                elif testtemp[k,j]==0:
                    labels[i,k,j,2]=1
                elif testtemp[k,j] <=sigtemp:
                    labels[i,k,j,3]=1
                else:
                    labels[i,k,j,4]=1



    return labels,totaldata

def main(seqsize,step,window):
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

    length = len(data)
    labels,totaldata = label_make(data[:length],diff[:length],sigma,seqsize,step,window)
    finaldates = dates[seqsize+1:]
    totaldata = np.array(totaldata,'float32')
    labels = np.array(labels,'uint8')


    pkl.dump([totaldata,labels,finaldates],open('/home/mharmon/FinanceProject/Data/tickdata/trainseq' + str(seqsize)+ 'win' + str(window) + '.pkl','wb'))


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)