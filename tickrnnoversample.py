# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense, RepeatVector, TimeDistributed
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import metrics
import time
from imblearn.over_sampling import SMOTE as smote
from keras.layers.wrappers import Bidirectional
import sys

def build_rnn(seqsize,batchsize):
    main_input = Input(shape=(seqsize,5,), dtype='float32', name='main_input')
    gru1 = GRU(512,return_sequences=True)(main_input)
    bn1 = BatchNormalization()(gru1)
    gru2 = GRU(512,return_sequences=True)(bn1)
    bn2 = BatchNormalization()(gru2)
    gru3 = GRU(512,return_sequences=False)(bn2)
    bn3 = BatchNormalization()(gru3)
    # I'm looping right here, but this may be overly complicated
    out = [Dense(5, activation='softmax', name=('main_output' + str(i)))(bn3) for i in range(5)]
    model = Model(main_input, out)

    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def sampling_from_sme(data,lab,seqsize):

    stocks = 5
    classes = 5
    neutral = 2
    totallabel = [[],[],[],[],[]]
    for i in range(stocks):

        templab = np.argmax(lab[i],axis=1)
        largestvec = np.where(templab==neutral)[0]
        tempdata = []
        templabel = [[],[],[],[],[]]
        for j in range(classes):

            if j!=neutral:
                othervec = np.where(templab==j)[0]


                if len(othervec)>0 and (len(othervec)/len(largestvec)<0.5):
                    # Here I need to find a label that occurs at least twice.  Any of them will do...
                    factor = 1
                    vals = np.zeros(len(othervec))

                    for m in range(stocks):
                        # Need to make a unique identifier for each that's quick to do.  Since the list are pretty small,
                        # I can just multiple by multiples of ten.  This will uniquely identify them.
                        if m!=i:
                            vals += factor*(np.argmax(lab[m],axis=1)[othervec]+1)
                            factor*=10

                    myuns = np.unique(vals)

                    # Find where there is a label that appears more than twice...
                    uniquelabels = []
                    smoteratio=[]
                    for m in range(len(myuns)):
                        onevec = np.where(myuns[m]==vals)[0]
                        if len(onevec)>=5:
                            uniquelabels +=[onevec]
                            smoteratio.append(len(onevec))
                    smoteratio = np.array(smoteratio)

                    smoteratio = 0.5*(smoteratio/np.sum(smoteratio))
                    if len(uniquelabels)>0:
                        for k in range(len(uniquelabels)):
                            myvec = uniquelabels[k]
                            uselabel = [lab[m][othervec[myvec[0]]] for m in range(5)]
                            tempvec = np.append(largestvec,np.array(othervec[myvec]))
                            fitlab = templab[tempvec]
                            fitlab[fitlab != neutral] = 0
                            fitlab[fitlab==neutral]=1
                            fitdat = np.reshape(data[tempvec],(len(data[tempvec]),seqsize*5))
                            sme = smote(ratio = smoteratio[k],random_state=42,k_neighbors=len(othervec[myvec])-1)
                            datanew,labnew = sme.fit_sample(fitdat, fitlab)

                            take = np.where(labnew ==0)[0]
                            datanew = datanew[take]
                            labnew = labnew[take]
                            takelen = len(data[myvec])

                            datanew = datanew[takelen:]
                            labnew = labnew[takelen:]
                            labnew += j


                            if len(tempdata)==0:
                                tempdata = datanew
                                for m in range(5):
                                        templabel[m] = np.array([uselabel[m],]*len(labnew))

                            else:
                                tempdata = np.vstack((tempdata,datanew))

                                for m in range(5):
                                    templabel[m]=np.vstack((templabel[m],np.array([uselabel[m],]*len(labnew))))


        if len(tempdata)>0:# Since data is all in one single stack
            if i==0:
                totaldata = tempdata
                for m in range(5):
                    totallabel[m] = templabel[m]
            else:
                totaldata = np.vstack((totaldata,tempdata))
                for m in range(5):
                    totallabel[m] = np.vstack((totallabel[m],templabel[m]))


    totaldata = np.reshape(totaldata,(len(totaldata),seqsize,5))
    totaldata = np.vstack((totaldata,data))

    for m in range(5):
        totallabel[m] = np.vstack((totallabel[m],lab[m]))


    # Don't forget that I'm going to have to mix the data up quite a bit as well.
    rng = np.random.randint(0,len(totaldata),len(totaldata))

    totaldata = totaldata[rng]

    for m in range(5):
        totallabel[m] = totallabel[m][rng]


    return totaldata,totallabel

# Train model here
def main(weekcount,ticknum):
    np.random.seed(13)
    mainadd = '/home/mharmon/FinanceProject//ModelResults/tick'+str(ticknum)
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train'+str(ticknum)+'.pkl'
    mydates = []
    batchsize=256
    data,labels,dates = pkl.load(open(address,'rb'))
    seqsize = ticknum
    model = build_rnn(seqsize,batchsize)
    month_len = 8064
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...


    #numtests = int(len(data)/week)
    numtests = 1
    f1testavg = np.zeros(numtests)
    modelsavepath = '/home/mharmon/FinanceProject/ModelResults/tick' + str(ticknum)+'/Models/tickmodel' + str(weekcount-1) + '.h5'

    beg = 0+week*weekcount
    end = month_len+week*weekcount
    pngcount=0
    f1store = np.zeros((numtests, 5, 5))
    stock1 = [[], [], [], [], []]
    stock2 = [[], [], [], [], []]
    stock3 = [[], [], [], [], []]
    stock4 = [[], [], [], [], []]
    stock5 = [[], [], [], [], []]
    f1wherestore=[]
    totalpred = []
    current_epoch=0

    # Load model if one is availabl

    if weekcount>0:
        model = load_model(modelsavepath)

        # Load the old f1 scores...
        f1save = mainadd+'/f1score.npy'
        f1avgsave = mainadd + '/f1avg.npy'
        fwheresave = mainadd + '/fwhere.pkl'
        f1store = np.load(f1save)
        f1testavg = np.load(f1avgsave)
        f1wherestore = pkl.load(open(fwheresave, 'rb'))


    for i in range(weekcount,numtests):
        modelsavepath = mainadd + '/Models/tickmodel' + str(i) + '.h5'
        epochx = data[beg:end]
        epoch_lab = labels[beg:end]
        testx = data[end:end+week]
        test_lab = labels[end:end+week]
        test_dates = dates[end:end+week]
        testy = [test_lab[:,i,:] for i in range(len(test_lab[0,:,0]))]
        # Do at least 5 validation sets here..

        vallen = int(len(epochx) * 0.15)

        trainx = epochx[0:len(epochx) - vallen]
        trainy = [epoch_lab[0:len(epochx) - vallen, i, :] for i in range(len(epoch_lab[0, :, 0]))]

        valx = epochx[len(epochx) - vallen:]
        valy = [epoch_lab[len(epochx) - vallen:, i, :] for i in range(len(epoch_lab[0, :, 0]))]


        for j in range(1):

            trainx, trainy = sampling_from_sme(trainx, trainy,seqsize)

            best_f1 = -1
            patience = 0
            best_loss = 10000
            while patience<5:
                current_epoch+=1
                firsttime = time.time()
                hist = model.fit(trainx, trainy, batch_size=batchsize, verbose=0, epochs=1, validation_data=(valx,valy))
                endtime = time.time()

                valpred = model.predict(valx,verbose=0)
                # Here is where I calculate f1 score to determine when to save my model...
                current_f1 = 0
                for f in range(5):
                    tempvalf1 = metrics.f1_score(y_true=np.argmax(valy[f], axis=1),
                                                   y_pred=np.argmax(valpred[f], axis=1), average=None)
                    current_f1+= np.average(tempvalf1)/5.


                #current_val = hist.history['val_loss'][0]
                print('')
                print('Window ' + str(i))
                print('Round ' + str(j))
                print('Epoch %r Took %.3f Seconds' % (current_epoch, (endtime - firsttime)))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                print('F1 Score is %.3f' %current_f1)

                # Capturing Best Loss Value
                current_loss = hist.history['loss'][0]
                if current_loss<best_loss:
                    best_loss = current_loss
                    print('Best Loss is %.6f' %best_loss)

                # Stopping Criterion
                if current_f1>best_f1:
                    best_f1 = current_f1
                    patience = 0
                    model.save(modelsavepath)
                    print('New Saved Model')

                    # Save model
                else:
                    patience+=1

            trainx = epochx[vallen:]
            trainy = [epoch_lab[vallen:, i, :] for i in range(len(epoch_lab[0, :, 0]))]

            valx = epochx[0:vallen]
            valy = [epoch_lab[0:vallen, i, :] for i in range(len(epoch_lab[0, :, 0]))]

            print()
            print('Loading Best Model For Next Round')
            print()

            del model
            model = load_model(modelsavepath)



        beg += week
        end += week

        predictions = model.predict(testx, batch_size=32, verbose=1)
        tempred = np.zeros((week,5,5))
        for p in range(5):
            tempred[:,p,:] =predictions[p]

        adds=mainadd + '/Predictions/predictions'+str(i)+'.pkl'
        pkl.dump([tempred,np.array(test_dates)],open(adds,'wb'))
        # This will need to be done for each softmax function
        # This gathers all of the f1 scores in a reasonable manner...
        for f in range(5):
            f1tempscore = metrics.f1_score(y_true=np.argmax(testy[f], axis=1),
                                           y_pred=np.argmax(predictions[f], axis=1), average=None)
            f1where = np.unique(np.argmax(testy[f], axis=1))
            f1wherestore += [f1where.tolist()]
            f1testavg[i]+= np.average(f1tempscore)/5.

            for c in range(len(f1where)):
                f1store[i, f, f1where[c]] = f1tempscore[c]

        f1save = mainadd + '/f1score.npy'
        f1avgsave = mainadd + '/f1avg.npy'
        fwheresave = mainadd + '/fwhere.pkl'
        np.save(f1save, np.array(f1store))
        np.save(f1avgsave,f1testavg)
        pkl.dump(f1wherestore,open(fwheresave,'wb'))
        # I should only have 25 histograms for all test runs
        print()
        print('Finished Window ' + str(i))
        print('Training Loss is ' + '%.5f' % hist.history['loss'][0])
        print('Validation Loss is ' + '%.5f' % hist.history['val_loss'][0])
        print('Best F1 is %.5f' %best_f1)

        # I need five histograms per week. It may be useful to also save the actual predictions along
        # with the histogram...
        for h in range(5):

            tempvec1 = np.where(np.argmax(testy[0], axis=1) == h)[0]
            tempvec2 = np.where(np.argmax(testy[1], axis=1) == h)[0]
            tempvec3 = np.where(np.argmax(testy[2], axis=1) == h)[0]
            tempvec4 = np.where(np.argmax(testy[3], axis=1) == h)[0]
            tempvec5 = np.where(np.argmax(testy[4], axis=1) == h)[0]

            if len(tempvec1) > 0:
                stock1[h] += predictions[0][tempvec1, h].tolist()
            if len(tempvec2) > 0:
                stock2[h] += predictions[1][tempvec2, h].tolist()
            if len(tempvec3) > 0:
                stock3[h] += predictions[2][tempvec3, h].tolist()
            if len(tempvec4) > 0:
                stock4[h] += predictions[3][tempvec4, h].tolist()
            if len(tempvec5) > 0:
                stock5[h] += predictions[4][tempvec5, h].tolist()

        # Print histograms at the end I suppose...
        figmain = mainadd + '/Figures/HistogramAllWeeksStock'

        for c in range(5):
            figfinal = figmain + '1Class' + str(c) + '.png'
            plt.figure(pngcount)
            plt.hist(stock1[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i + 1) + ' Weeks For Stock 1' + ' And Class ' + str(c))
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

if __name__=='__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    main(weekcount,ticknum)