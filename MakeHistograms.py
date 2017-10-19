# Author: Mark Harmon
# Purpose: Make histograms for the finance presentation

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def main():
    addtemp = '/home/mharmon/FinanceProject/ModelResults/conv1d/Predictions/predictions'
    datadd = '/home/mharmon/FinanceProject/Data/tickdata/train10cnn0.pkl'

    data,lab,dates = pkl.load(open(datadd,'rb'))

    del data
    del dates

    month_len = 8064
    week = int((month_len)/4.)

    end = month_len

    # Do this for each week
    stock1 = [[], [], [], [], []]
    stock2 = [[], [], [], [], []]
    stock3 = [[], [], [], [], []]
    stock4 = [[], [], [], [], []]
    stock5 = [[], [], [], [], []]
    for i in range(26):

        testlab = lab[end:end+week]
        testlab = [testlab[:,i,:] for i in range(len(testlab[0,:,0]))]
        predadd = addtemp + str(i) + '.pkl'
        predictions,dates = pkl.load(open(predadd,'rb'))

        end = end + week
        for h in range(5):

            tempvec1 = np.where(np.argmax(testlab[0], axis=1) == h)[0]
            tempvec2 = np.where(np.argmax(testlab[1], axis=1) == h)[0]
            tempvec3 = np.where(np.argmax(testlab[2], axis=1) == h)[0]
            tempvec4 = np.where(np.argmax(testlab[3], axis=1) == h)[0]
            tempvec5 = np.where(np.argmax(testlab[4], axis=1) == h)[0]

            if len(tempvec1) > 0:
                stock1[h] += predictions[tempvec1,0, h].tolist()
            if len(tempvec2) > 0:
                stock2[h] += predictions[tempvec2,1, h].tolist()
            if len(tempvec3) > 0:
                stock3[h] += predictions[tempvec3,2, h].tolist()
            if len(tempvec4) > 0:
                stock4[h] += predictions[tempvec4,3, h].tolist()
            if len(tempvec5) > 0:
                stock5[h] += predictions[tempvec5,4, h].tolist()

    # Now for each stock
    figmain = '/home/mharmon/FinanceProject/ModelResults/HistogramAllWeeksStock'
    pngcount = 0
    for c in range(5):
        figfinal = figmain + '1Class' + str(c) + '.png'
        plt.figure(pngcount)
        plt.hist(stock1[c])
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 1' + ' And Class ' + str(c))
        if c==1 or c==3:
            plt.ylim(ymax=2500, ymin=0)
        plt.savefig(figfinal)
        plt.close()
        pngcount += 1

        figfinal = figmain + '2Class' + str(c) + '.png'
        plt.figure(pngcount)
        plt.hist(stock2[c])
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 2' + ' And Class ' + str(c))
        plt.savefig(figfinal)
        plt.close()
        pngcount += 1

        figfinal = figmain + '3Class' + str(c) + '.png'
        plt.figure(pngcount)
        plt.hist(stock3[c])
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 3' + ' And Class ' + str(c))
        plt.savefig(figfinal)
        plt.close()
        pngcount += 1

        figfinal = figmain + '4Class' + str(c) + '.png'
        plt.figure(pngcount)
        plt.hist(stock4[c])
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 4' + ' And Class ' + str(c))
        plt.savefig(figfinal)
        plt.close()
        pngcount += 1

        figfinal = figmain + '5Class' + str(c) + '.png'
        plt.figure(pngcount)
        plt.hist(stock5[c])
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 5' + ' And Class ' + str(c))
        plt.savefig(figfinal)
        plt.close()
        pngcount += 1


    return

if __name__ == '__main__':
    main()