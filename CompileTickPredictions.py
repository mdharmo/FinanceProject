# Author: Mark Harmon
# Purpose: This is where I compile the tick predictions for Diego's guy...

import pickle as pkl
import numpy as np

from os import listdir
from os.path import isfile, join

def main():
    mypath = '/home/mharmon/FinanceProject/ModelResults/ticknewday10win1/Predictions/'

    for j in range(4):

        starting_hour = 11
        if j==0:
            filerange=56+1
        if j==2:
            filerange = 61+1
        if j==3 or j==1:
            filerange = 61+1

        for i in range(filerange):
            load_path = mypath +'Sector'+str(j) +'predictions' + str(i) + 'run0'+'.pkl'
            temppred,tempdates = np.load(load_path)



            if i ==0 and j==0:
                pred = temppred
                dates = tempdates
            else:
                pred = np.vstack((pred,temppred))
                dates = np.concatenate((dates,tempdates))

    add = '/home/mharmon/FinanceProject/Data/tickdata/returnsForDiego.csv'
    data = np.loadtxt(open(add,'rb'),delimiter=',')

    actual_dates = np.zeros((len(data[2016*4+23:-861]),3))
    actual_dates[:,0] = np.copy(data[2016*4+23:-861,0])
    actual_dates[:,1] = np.copy(data[2016*4+23:-861,1])
    actual_dates[:,2] = np.copy(data[2016*4+23:-861,2])


    pred = np.reshape(pred,(len(pred),22,5))

    #stock = ['oil','eurusd','gold','mini','ust']
    for i in range(22):
        savepath = '/home/mharmon/FinanceProject/ModelResults/ticknewday10win1/predfund'+str(i+1)+'.csv'
        np.savetxt(savepath, pred[:,i,:],delimiter=',')

    np.savetxt('/home/mharmon/FinanceProject/ModelResults/tickcorrectday10win1/dates.csv',actual_dates,delimiter=',')
    return

if __name__=='__main__':
    main()