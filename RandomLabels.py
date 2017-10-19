# Author: Mark Harmon
# Purpose: Make labels and final trading data for input into recurrent model for tick data

# This is to make my training set and labels for the cnn sequence.  I'm going to treat it similarly to a video problem
# by having one sequence == one image.  This model should inherently be better than my current recurrent model...
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys

def label_make(data,diff,sigma,seqsize,num_seq,step,window):
    # 5 stocks and 5 labels...
    changelen=4
    lablenint = int((len(diff)-(seqsize+ num_seq*changelen))/step)
    lablenfloat = (len(data)-(seqsize+ num_seq*changelen))/float(step)
    test_num = int((window-step-(lablenfloat-lablenint)*step)/step)
    lablen = lablenint - np.max((0,test_num))
    labels = np.zeros((lablen,window,5,5))

    beg = seqsize-1
    totaldata = []
    labbeg = seqsize + num_seq*changelen-1
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
            dattemp += [data[i*step + k*changelen:beg+i*step+1+k*changelen,:].T.tolist()]

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

                if testtemp[k,j]<-2*sigtemp:
                    labels[i,k,j,0]=1
                elif testtemp[k,j]<-sigtemp:
                    labels[i,k,j,1]=1
                elif testtemp[k,j]<=sigtemp:
                    labels[i,k,j,2]=1
                elif testtemp[k,j] <=2*sigtemp:
                    labels[i,k,j,3]=1
                else:
                    labels[i,k,j,4]=1

    return labels,totaldata

def main(seqsize,step,window):
    num_seq = 4
    changelen = 4

    diff_dist = int(2016/4.)
    num_dists = 400

    sig_dist = np.random.uniform(low=0,high=0.01,size=(num_dists,5))
    mean_dist = np.random.uniform(low=-1,high=1,size=(num_dists,5))
    a = np.linspace(0,5,num_dists)
    for i in range(5):
        mean_dist[:,i]+=a
    beg=0
    end=diff_dist
    data = np.zeros((diff_dist*num_dists,5))
    for i in range(num_dists):
        for j in range(5):
            data[beg:end,j]=np.random.normal(loc=mean_dist[i,j],scale=sig_dist[i,j],size=diff_dist)
        beg=end
        end+=diff_dist


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
    myrange = int(len(data) / 1.)
    beg = 0
    end = myrange

    sigsave = '/home/mharmon/FinanceProject/Data/tickdata/sigma.pkl'
    pkl.dump(sigma,open(sigsave,'wb'))

    labels,totaldata = label_make(data[beg:end],diff[beg:end],sigma,seqsize,num_seq,step,window)
    totaldata = np.array(totaldata,'float32')
    totaldata = np.reshape(totaldata,(len(totaldata),num_seq,1,5,seqsize))
    labels = np.array(labels,'uint8')

    datasave = '/home/mharmon/FinanceProject/Data/tickdata/randomtrain' + str(seqsize)+'cnn' + str(i) + '.pkl'
    pkl.dump([totaldata,labels],open(datasave,'wb'))


    # Let's look at the distributions...

    pngcount=0
    figsave = '/home/mharmon/FinanceProject/fig'
    for i in range(5):
        finalsave = figsave + str(i) + '.png'
        plt.figure(pngcount)
        plt.plot(data[:,i],linewidth=2)
        plt.title('Stock 1')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig(finalsave)
        plt.close()
        pngcount+=1

    return

if __name__=='__main__':
    seqsize = int(sys.argv[1])
    step = int(sys.argv[2])
    window = int(sys.argv[3])
    main(seqsize,step,window)