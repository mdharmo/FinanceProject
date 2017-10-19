# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# This is to make my training set and labels for the cnn sequence.  I'm going to treat it similarly to a video problem
# by having one sequence == one image.  This model should inherently be better than my current recurrent model...


import numpy as np
import pickle as pkl
import sys

def label_make(diff,sigma,seqsize,num_seq,step,window):
    # 5 stocks and 5 labels...
    changelen=4

    lablenint = int((len(diff)-(seqsize+ (num_seq-1)*changelen))/step)
    labels = np.zeros((lablenint,window,5,5))

    beg = seqsize
    totaldata = []
    labbeg = seqsize + (num_seq-1)*changelen
    factor = 2
    week_len = 2016
    # For when I do the day type label creation
    day_len = int(week_len/7)
    stocks=5
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
    address = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'
    num_seq = 4
    changelen = 4
    data,dates = pkl.load(open(address,'rb'))
    diff = np.zeros((len(data)-1,5))

    for i in range(len(data)-1):
        diff[i,:] = data[i+1,:]-data[i,:]

    for i in range(5):
        diff[:,i] = (diff[:,i]-np.mean(diff[:,i]))/np.std(diff[:,i])
    # Calculate my sigma values

    for i in range(5):
        diff[:,i] = (diff[:,i]-np.min(diff[:,i]))/(np.max(diff[:,i])-np.min(diff[:,i]))

    sigma = np.zeros(5)
    for i in range(5):
        sigma[i] = np.std(diff[:,i])

    # Now make the actual labels
    myrange = int(len(data) / 4.)
    beg = 0
    end = myrange

    sigsave = '/home/mharmon/FinanceProject/Data/tickdata/sigma.pkl'
    pkl.dump(sigma,open(sigsave,'wb'))
    for i in range(4):

        # Instead of using data, I should be using diff as my actual data..
        labels,totaldata = label_make(diff[beg:end],sigma,seqsize,num_seq,step,window)
        finaldates = dates[beg+seqsize+(num_seq-1)*changelen+1:end+1]
        totaldata = np.array(totaldata,'float32')
        totaldata = np.reshape(totaldata,(len(totaldata),num_seq,1,5,seqsize))
        labels = np.array(labels,'uint8')

        datasave = '/home/mharmon/FinanceProject/Data/tickdata/trainday' + str(seqsize)+'win' + str(window) +'cnn'+str(i)+ '.pkl'
        pkl.dump([totaldata,labels,finaldates],open(datasave,'wb'))


        beg = end-(seqsize +changelen*(num_seq-1))
        end = beg + myrange


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)