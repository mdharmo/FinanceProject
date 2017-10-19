# Author: Mark Harmon
# Purpose: Weekly Averages for each stock figures of the tick data


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def main():

    add = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'

    data,chunk = pkl.load(open(add,'rb'))
    del chunk

    week_len = 2016

    x = np.linspace(0,int(len(data)/float(week_len)))

    averages = np.zeros((len(x),5))

    for i in range(len(x)):

        for k in range(5):
            averages[i,k] = np.mean(data[i*week_len:(i+1)*week_len,k])


    stock = ['Oil', 'EURUSD', 'Gold', 'MINI', 'UST']
    pngcount=0
    saveadd = '/home/mharmon/FinanceProject/ModelResults/SpecialFigures/'
    for k in range(5):

        plt.figure(pngcount)
        plt.plot(x,averages[:,k],linewidth=2)
        plt.xlabel('Week')
        plt.ylabel('Average Value Over the Week')
        plt.title('Weekly Average For ' + stock[k])
        figsave = saveadd + stock[k] + 'WeekAverage.png'
        plt.savefig(figsave)
        plt.close()
        pngcount+=1


    return

if __name__=='__main__':

    main()