# Author: Mark Harmon
# Purpose: We have the processed data as a python list.  However, these need to converted to numpy files to be used
# in a neural network.
import csv
import numpy as np
import pickle as pkl
# Load the data here:
instrumentdict = dict()
keyvalue = 1

for j in range(5):

    if j+7<10:
        address = '/home/mharmon/FinanceProject/Data/cmecsv/cmebook0' + str(j+7) + '.csv'
    else:
        address = '/home/mharmon/FinanceProject/Data/cmecsv/cmebook' + str(j + 7) + '.csv'

    f = open(address)

    # First we put everything into a list from the csv file....
    reader = csv.reader(f,delimiter=',')
    data=[]
    for row in reader:
        data+=[row]


    # Now the difficult part is actually processing the data.  I'm probably going to have to concat with numpy for each
    # Here is my initial matrix for my data.

    # The data values that I'm going to keep are as follows:
    # Time sent, time received, bid orders, bid order quantity, bid price, askprice, ask order quantity, ask orders, repeat
    # for level 2 of the order book from bidorders:askorders
    bookdata=[]
    for i in range(len(data)):

        # First check is to make sure I have enough proper data.
        if len(data[i])>=15:

            # Then I check to see if there are None values in the data.  If there are
            # ignore them...

            tempbookdata = data[i][1:15]

            if not any(np.array(tempbookdata)=='None'):

                # I make a dictionary to keep track of the different instruments that I've encountered.
                # This must remain constant while running through all of my data...
                key = data[i][0]
                if key not in instrumentdict:
                    instrumentdict[key] = keyvalue
                    keyvalue += 1
                    print(key)
                    print(keyvalue)

                mine = str(instrumentdict[key])
                tempbookdata.append(mine)

                if bookdata == []:
                    bookdata = [tempbookdata]
                else:
                    bookdata += [tempbookdata]

    # At the end, we save the data to a pickle file

    dataname2 = '/home/mharmon/FinanceProject/Data/cmedata/cmebook0' + str(j + 7) + 'list.pkl'
    pkl.dump(bookdata,open(dataname2,'wb'))
    bookdata=[]

# Save the dictionary at the very end
dictname = '/home/mharmon/FinanceProject/Data/cmedata/namedict.pkl'
pkl.dump(instrumentdict,open(dictname,'wb'))
