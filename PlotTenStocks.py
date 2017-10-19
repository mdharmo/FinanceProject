import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def main():

    day_len = 288
    add = '/home/mharmon/FinanceProject/Data/tickdata/returnsForDiego.csv'

    stuff = np.loadtxt(open(add,'rb'),delimiter=',')
    data = stuff[:,3:]

    # Now I just want to play 30 two day intervals for 10 stocks

    main_add = '/home/mharmon/FinanceProject/ModelResults/StockFigures/s'
    # For each sotck
    for i in range(10):

        # For each two day period. Start after the first day...
        beg = 288
        end = beg + 2*day_len
        for j in range(30):
            savefig = main_add + str(i+1)+'/Figure'+str(j)+'.png'
            plt.plot(data[beg:end,i],linewidth=2)
            plt.title('Two Day Plot')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.savefig(savefig)
            plt.close()
            beg=end
            end=beg+2*day_len





    return

if __name__=='__main__':

    main()