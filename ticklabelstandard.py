# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data


import numpy as np
import pickle as pkl

def label_make(data,diff,sigma,seqsize):
    # 5 stocks and 5 labels...
    labels = np.zeros((len(data)-seqsize-1,seqsize,5,5),'uint8')

    beg = seqsize-1
    totaldata = []
    for i in range(len(labels)):


        totaldata += [data[i:beg+i+1,:].tolist()]

        for k in range(seqsize):
            testtemp = diff[i+k, :]
            for j in range(5):

                sigtemp = sigma[j]

                if testtemp[j]<-sigtemp:
                    labels[i,k,j,0]=1
                elif testtemp[j]<0:
                    labels[i,k,j,1]=1
                elif testtemp[j]==0:
                    labels[i,k,j,2]=1
                elif testtemp[j] <=sigtemp:
                    labels[i,k,j,3]=1
                else:
                    labels[i,k,j,4]=1

    return labels,totaldata

def main():
    address = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'
    seqsize = 20
    data,dates = pkl.load(open(address,'rb'))
    diff = np.zeros((len(data)-1,5))

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
    lensplit = int(len(diff)/8.)

    beg = 0
    end = lensplit
    sigsave = '/home/mharmon/FinanceProject/Data/tickdata/sigma.pkl'
    pkl.dump(sigma,open(sigsave,'wb'))
    for i in range(8):
        labels,totaldata = label_make(data[beg:end],diff[beg:end],sigma,seqsize)
        finaldates = dates[beg+1:end+1]
        totaldata = np.array(totaldata,'float32')
        labels = np.array(labels,'uint8')
        add = '/home/mharmon/FinanceProject/Data/tickdata/train20standard' + str(i)+'.pkl'
        pkl.dump([totaldata,labels,finaldates],open(add,'wb'))


    return

if __name__=='__main__':
    main()