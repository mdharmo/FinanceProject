# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# So, let's make the actual training data as well.

import numpy as np
import pickle as pkl
import sys

def label_make(data,diff,sigma,seqsize,step,window):
    # 5 stocks and 5 labels...
    # Binary version is a single label...
    lablenint = int((len(data)-seqsize-1)/step)
    lablenfloat = (len(data)-seqsize-1)/float(step)
    lablen = lablenint - np.max(0,int((window-step-1-(lablenfloat-lablenint)*step)/step))
    labels = np.zeros((lablen,window,5,1))

    beg = seqsize-1
    totaldata = []
    for i in range(len(labels)):

        testtemp = diff[beg+i*step:beg+i*step+window,:]
        totaldata += [data[i*step:beg+i*step+1,:].tolist()]

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


                if testtemp[k,j]==0:
                    labels[i,k,j,0]=0
                else:
                    labels[i,k,j,0]=1

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
    pkl.dump([totaldata,labels,finaldates],open('/home/mharmon/FinanceProject/Data/tickdata/trainbinseq' + str(seqsize)+ 'win' + str(window) + '.pkl','wb'))


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)