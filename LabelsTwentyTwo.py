# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# This is to make my training set and labels for the cnn sequence.  I'm going to treat it similarly to a video problem
# by having one sequence == one image.  This model should inherently be better than my current recurrent model...


import numpy as np
import pickle as pkl
import sys

def label_make(diff,sigma,seqsize,num_seq,step,window,stocks):
    # 5 stocks and 5 labels...
    changelen=4

    lablenint = int((len(diff)-(seqsize+ (num_seq-1)*changelen+window))/step)
    labels = np.zeros((lablenint,window,stocks,5))

    beg = seqsize
    totaldata = []
    labbeg = seqsize + (num_seq-1)*changelen
    factor = 2
    week_len = 2016
    # For when I do the day type label creation
    day_len = int(week_len/7)
    week_count = 1

    for i in range(len(labels)):

        testtemp = diff[labbeg + i * step:labbeg + i * step + window, :]
        dattemp = []
        for k in range(num_seq):
            dattemp += [diff[i*step + k*changelen:beg+i*step+k*changelen,:].T.tolist()]

        totaldata += [dattemp]

        if beg+i*step+window>factor*day_len:
            week_count+=1
            factor+=1
        # Need to calculate a new sigma value after each week
        sigma = np.zeros(stocks)
        for m in range(stocks):
            sigma[m] = np.std(diff[(week_count-1)*day_len:week_count*day_len])

        for k in range(window):
            for j in range(stocks):

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
    address = '/home/mharmon/FinanceProject/Data/tickdata/returnsForDiego.csv'
    num_seq = 4
    changelen = 4
    data= np.loadtxt(open(address,'rb'),delimiter=',')
    dates = np.copy(data[:,0])
    data = data[:,3:25]
    stocks =22
    diff = data[1:]

    '''
    for i in range(22):
        diff[:,i] = (diff[:,i]-np.mean(diff[:,i]))/np.std(diff[:,i])
    '''
    # Calculate my sigma values
    sigma = np.zeros(stocks)
    for i in range(stocks):
        sigma[i] = np.std(diff[:,i])

    # Now make the actual labels
    myrange = int(len(data) / 4.)
    beg = 0
    end = myrange

    sigsave = '/home/mharmon/FinanceProject/Data/tickdata/sigma.pkl'
    pkl.dump(sigma,open(sigsave,'wb'))
    for i in range(4):

        # Instead of using data, I should be using diff as my actual data..
        labels,totaldata = label_make(diff[beg:end],sigma,seqsize,num_seq,step,window,stocks)
        finaldates = dates[beg+seqsize+(num_seq-1)*changelen+1:end+1]
        totaldata = np.array(totaldata,'float32')
        totaldata = np.reshape(totaldata,(len(totaldata),num_seq,1,stocks,seqsize))
        labels = np.array(labels,'uint8')

        datasave = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday' + str(seqsize)+'win' +str(window)+'sector'+ str(i) + '.pkl'
        pkl.dump([totaldata,labels,finaldates],open(datasave,'wb'))


        beg = end-seqsize + (num_seq-1)*changelen
        end = beg + myrange


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)