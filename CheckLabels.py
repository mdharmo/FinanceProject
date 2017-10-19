# Author: Mark Harmon
# Purpose: I use this code to check that my labels are current for the stock project...

import pickle as pkl
import numpy as np

address = '/home/mharmon/FinanceProject/Data/cmenew/cmebook70.pkl'

data,labels = pkl.load(open(address,'rb'))

labelrange = 50
diff = []
for i in range(int(len(data)/50)-1):

    s2temp = data[50*(i+1)-1,63]-data[(i+1)*50-1,62]
    s1temp = data[i*50-1,63]-data[i*50-1,62]

    diff+=[s2temp-s1temp]



spread1 = np.zeros(38)
spread2 = np.zeros(38)

for i in range(38):
    spread1[i] = data[-1,12*i+3]-data[-1,12*i+2]
    spread2[i] = data2[49,12*i+3]-data2[49,12*i+2]

