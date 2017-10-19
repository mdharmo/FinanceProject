import scanf as s
import csv
import numpy as np

InputFileName = "/home/mharmon/FinanceProject/Data/2015-12-11-mdp_book_builder_output.log"
f = open(InputFileName)
lines = f.readlines()
orderlevels = np.zeros(10)
count = 0

for i in range(len(lines)):

    if lines[i][0]=='(':
        count+=1
    else:


        if count==2:
            orderlevels[count-1]+=1

        if count==3:
            orderlevels[count-1]+=1

        if count==4:
            orderlevels[count-1]+=1
        if count==5:
            orderlevels[count-1]+=1
        if count==6:
            orderlevels[count-1]+=1

        if count==7:
            orderlevels[count-1]+=1
        if count==8:
            orderlevels[count-1]+=1
        if count==9:
            orderlevels[count-1]+=1
        if count==10:
            orderlevels[count-1]+=1

        count =0
