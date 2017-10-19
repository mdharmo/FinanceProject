# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# This is to make my training set and labels for the cnn sequence.  I'm going to treat it similarly to a video problem
# by having one sequence == one image.  This model should inherently be better than my current recurrent model...


import numpy as np
import pickle as pkl

def label_make(data,diff,sigma,seqsize,num_seq):
    # 5 stocks and 5 labels...
    changelen = 4
    labels = np.zeros((len(data)-(seqsize + num_seq*changelen)-1,5,5))
    beg = seqsize-1
    totaldata = []
    labbeg = seqsize + num_seq*changelen-1
    for i in range(len(labels)):

        testtemp = diff[labbeg+i,:]

        dattemp = []
        for k in range(num_seq):
            dattemp += [data[i + k*changelen:beg+i+1+k*changelen,:].T.tolist()]

        totaldata += [dattemp]
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

def main():
    address = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'
    seqsize = 10
    num_seq = 4
    changelen = 4
    data,dates = pkl.load(open(address,'rb'))
    diff = np.zeros((len(data)-1,5))

    for i in range(5):
        data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

    # Calculate differences
    for i in range(len(data)-1):
        diff[i,:] = data[i+1,:]-data[i,:]

    # Calculate my sigma values
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

        labels,totaldata = label_make(data[beg:end],diff[beg:end],sigma,seqsize,num_seq)
        finaldates = dates[beg+seqsize+num_seq*changelen+1:end+1]
        totaldata = np.array(totaldata,'float32')
        totaldata = np.reshape(totaldata,(len(totaldata),num_seq,1,5,seqsize))
        labels = np.array(labels,'uint8')

        datasave = '/home/mharmon/FinanceProject/Data/tickdata/train10cnn' + str(i) + '.pkl'
        pkl.dump([totaldata,labels,finaldates],open(datasave,'wb'))
        beg = end
        end = beg + myrange


    return

if __name__=='__main__':
    main()