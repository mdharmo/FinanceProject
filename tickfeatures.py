# Author: Mark Harmon
# Purpose: Make the tick features usable for a stateful neural network...


import numpy as np
import pickle as pkl
import os

def load_data(address,datadd,dayadd):

    data = np.loadtxt(address+datadd,delimiter=',')
    day = np.loadtxt(address + dayadd, delimiter=',')

    return data,day
def main(mainadd):

    dirs = os.listdir(mainadd)

    # First we gather the data and get the beginning date...
    largestfirstdate = 0.
    totaldata = []
    totalday = []
    for i in range(len(dirs)):

        address = '/home/mharmon/FinanceProject/Data/tickdata/'+dirs[i]+'/'
        datadd = 'data0.dat'
        dayadd = 'date0.dat'
        datatemp,daytemp = load_data(address,datadd,dayadd)

        if largestfirstdate<daytemp[0]:
            largestfirstdate = daytemp[0]

        print(i)
        totaldata+=[datatemp.tolist()]
        totalday+=[daytemp.tolist()]


    # Here is where we cut the length
    finaldata = []
    for i in range(len(dirs)):

        datatemp = np.array(totaldata[i])
        daytemp = np.array(totalday[i])

        vecdelete = np.where(largestfirstdate==daytemp)[0]

        datatemp = np.delete(datatemp,np.arange(vecdelete[0]))

        finaldata += [datatemp.tolist()]

    finaldata = np.array(finaldata,'float32').T
    a = np.where(totalday[0]==largestfirstdate)[0]
    finaldates = np.array(totalday[0][a[0]:])


    pkl.dump([finaldata,finaldates],open('/home/mharmon/FinanceProject/Data/tickdata/traindata.pkl','wb'))

    # Really, I we want is the tick data since the hour/min/date stuff we all be the same.  What we have to check for
    # though, is that we have the same length

if __name__=='__main__':
    mainadd='/home/mharmon/FinanceProject/Data/tickdata/'
    main(mainadd)
