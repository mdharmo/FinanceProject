# Author: Mark Harmon
# Purpose: This is to look at the non-stationarity of my data.

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def main():
    main_add = '/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl'
    save_main = '/home/mharmon/FinanceProject/ModelResults/ExpFigures/'
    data,dates = pkl.load(open(main_add,'rb'))
    stocks = 5
    pngcount = 0
    for i in range(stocks):
        tempstock1 = data[1:,i]
        tempstock2 = data[0:-1,i]

        price_diff = tempstock1-tempstock2

        plt.figure(pngcount)
        plt.plot(price_diff[0:100000],linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Stock ' + str(i+1) + ' Price Differences')
        save_fig = save_main + 'Stock'+str(i+1)+'.png'
        plt.savefig(save_fig)
        plt.close()


    return

if __name__=='__main__':

    main()