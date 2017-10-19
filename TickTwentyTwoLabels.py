# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# This is to make my training set and labels for the cnn sequence.  I'm going to treat it similarly to a video problem
# by having one sequence == one image.  This model should inherently be better than my current recurrent model...


import numpy as np
import pickle as pkl
import sys

def label_make(diff,seqsize,step,window,stocks):

    classes = 5
    week_len = 2016
    day_len = int(week_len/7)
    # 22 stocks and 5 labels...
    labels = np.zeros((len(diff)-seqsize-1-window,window,stocks,classes))

    beg = seqsize-1
    totaldata = []
    week_count = 1
    factor = 2
    for i in range(len(labels)):

        testtemp = diff[beg + i * step:beg + i * step + window,:]
        totaldata += [diff[i:beg+i+1,:].tolist()]

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
    data= np.loadtxt(open(address,'rb'),delimiter=',')
    dates = np.copy(data[:,0])
    data = data[:,3:25]
    stocks = 22
    diff = data[1:]

    # Instead of using data, I should be using diff as my actual data..
    labels,totaldata = label_make(diff,seqsize,step,window,stocks)
    finaldates = dates[seqsize+1:]
    totaldata = np.array(totaldata,'float32')
    labels = np.array(labels,'uint8')

    datasave = '/home/mharmon/FinanceProject/Data/tickdata/train' + str(seqsize)+'win' +str(window)+'.pkl'
    pkl.dump([totaldata,labels,finaldates],open(datasave,'wb'))


    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)