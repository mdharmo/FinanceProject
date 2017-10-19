# Author: Mark Harmon
# Purpose: To check

import matplotlib
matplotlib.use('pdf')
import pickle as pkl
import numpy as np
from scipy import stats

def main():

    stocks = 5

    for i in range(6):
        add = '/home/mharmon/FinanceProject/Data/tickdata/trainseq30win' + str(5+i) + '.pkl'
        data,labels,dates = pkl.load(open(add,'rb'))
        print(labels.shape)
        del data
        del dates


        # For each stock calculate equality among windows
        window_size = 5+i

        real_count=0
        for j in range(len(labels)):

            checker =1
            for k in range(window_size-1):

                if np.all(labels[j,k,:,:]==labels[j,k+1,:,:]):
                    checker+=1
            if checker==5+i:
                real_count+=1



        real_count /=float(len(labels))
        real_count*=100
        print('The all percentage is')
        print('%r' %real_count)
        print('')








    return

if __name__=='__main__':
    main()