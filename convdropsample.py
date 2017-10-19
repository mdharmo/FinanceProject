# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense, RepeatVector, TimeDistributed, Flatten, Permute, merge,Lambda,Reshape
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import metrics
import time
from imblearn.over_sampling import SMOTE as smote
import sys
from SmoteSampling import smote_conv_seq
from keras.layers import Activation
from RegConvLSTM import ConvLSTM2D

def build_rnn(seqsize,batchsize,labseq):
    classes = 5
    # Build Encoder here
    main_input = Input(shape=(4,5,seqsize,1), dtype='float32', name='main_input')

    # This is where we do attention (This may need to change, attention for conv models seems odd).
    a1 = Reshape((4,5*seqsize))(main_input)
    a2 = TimeDistributed(Dense(1, activation='tanh'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(5*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,5,seqsize,1))(a6)
    att = merge([main_input, a7],mode='mul')

    # Encoder first
    conv1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True,dropout=0.2)(att)
    bn1 = BatchNormalization()(conv1)
    conv2 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=False,dropout=0.2)(bn1)
    bn2 = BatchNormalization()(conv2)

    # Decoder here

    rep0 = Flatten()(bn2)
    rep1 = RepeatVector(labseq)(rep0)
    rep2 = Reshape((labseq,5,seqsize-2,64))(rep1)
    dec1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True,dropout=0.2)(rep2)
    bn3 = BatchNormalization()(dec1)
    res = Reshape((labseq,5*(seqsize-3)*64))(bn3)

    out = [TimeDistributed(Dense(5, activation='softmax', name=('predict_out' + str(i))))(res) for i in range(5)]

    # Now we combine the confidence output and prediction output into a single output...
    model = Model(main_input,out)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# Train model here
def main(weekcount,ticknum,winnum):
    labseq = winnum
    stocks = 5
    classes = 5

    mainadd = '/home/mharmon/FinanceProject/ModelResults/tickconvdrop'+str(ticknum) + 'win' + str(winnum)
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train'+str(ticknum) + 'cnn0.pkl'
    batchsize=256
    data,labels,dates = pkl.load(open(address,'rb'))
    seqsize = ticknum
    model = build_rnn(seqsize,batchsize,labseq)
    month_len = int(8064)
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...


    #numtests = int(len(data)/week)
    numtests = 30
    pred_runs = 30
    f1testavg = np.zeros((numtests,winnum))
    modelsavepath = mainadd + '/Models/tickmodel' + str(np.max((0,weekcount-1))) + '.h5'

    beg = 0+week*weekcount
    end = month_len+week*weekcount
    pngcount=0
    f1store = np.zeros((numtests, 5, winnum, 5))
    f1wherestore=[]
    totalpred = []

    data = np.swapaxes(data,2,4)
    data = np.swapaxes(data,2,3)

    # Load model if one is availabl

    if weekcount>0:
        model.load_weights(modelsavepath)

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
        testy = [test_lab[:,:,i,:] for i in range(len(test_lab[0,0,:,0]))]
        # Do at least 5 validation sets here..

        vallen = int(len(epochx) * 0.15)

        trainx = epochx[0:len(epochx) - vallen]
        trainy = [epoch_lab[0:len(epochx) - vallen,:, i, :] for i in range(len(epoch_lab[0,0, :, 0]))]

        valx = epochx[len(epochx) - vallen:]
        valy = [epoch_lab[len(epochx) - vallen:,:, i, :] for i in range(len(epoch_lab[0, 0,:, 0]))]


        for j in range(2):

            trainx, trainy = smote_conv_seq(trainx, trainy,seqsize,winnum,stocks,classes)

            best_f1 = -1
            patience = 0
            while patience<5:
                firsttime = time.time()
                hist = model.fit(trainx, trainy, batch_size=batchsize, verbose=0, epochs=1, validation_data=(valx,valy))
                endtime = time.time()

                valpred = model.predict(valx,verbose=0)
                # Here is where I calculate f1 score to determine when to save my model...
                current_f1 = np.zeros((winnum))
                for ti in range(winnum):
                    for f in range(stocks):
                        tempvalf1 = metrics.f1_score(y_true=np.argmax(valy[f][:,ti,:], axis=1),
                                                       y_pred=np.argmax(valpred[f][:,ti,:], axis=1), average=None)
                        current_f1[ti]+= np.average(tempvalf1)/(stocks)


                #current_val = hist.history['val_loss'][0]
                print('')
                print('Window ' + str(i))
                print('Round ' + str(j))
                print('Epoch Took %.3f Seconds' % (endtime - firsttime))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                print('F1 Score is %.3f' %np.mean(current_f1))
                if np.mean(current_f1)>best_f1:
                    best_f1 = np.mean(current_f1)
                    patience = 0
                    model.save_weights(modelsavepath)
                    print('New Saved Model')

                    # Save model
                else:
                    patience+=1

            trainx = epochx[vallen:]
            trainy = [epoch_lab[vallen:,:, i, :] for i in range(len(epoch_lab[0,0, :, 0]))]

            valx = epochx[0:vallen]
            valy = [epoch_lab[0:vallen,:, i, :] for i in range(len(epoch_lab[0,0, :, 0]))]

            print()
            print('Loading Best Model For Next Round')
            print()
            model.load_weights(modelsavepath)



        beg += week
        end += week

        for r in range(pred_runs):

            predictions = model.predict(testx, batch_size=32, verbose=1)
            tempred = np.zeros((week,5,winnum,5))
            for p in range(5):
                tempred[:,p,:,:] =predictions[p]


            adds=mainadd + '/Predictions/predictions'+str(i)+'run'+str(r)+'.pkl'
            pkl.dump([tempred,np.array(test_dates)],open(adds,'wb'))
        # This will need to be done for each softmax function
        # This gathers all of the f1 scores in a reasonable manner...
        for ti in range(winnum):
            for f in range(stocks):
                f1tempscore = metrics.f1_score(y_true=np.argmax(testy[f][:,ti,:], axis=1),
                                               y_pred=np.argmax(predictions[f][:,ti,:], axis=1), average=None)
                f1where = np.unique(np.argmax(testy[f][:,ti,:], axis=1))
                f1wherestore += [f1where.tolist()]
                f1testavg[i,ti] += np.average(f1tempscore)/(stocks*winnum)

                for c in range(len(f1where)):
                    f1store[i, f, ti,f1where[c]] = f1tempscore[c]

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
        '''
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
        '''
    return

if __name__=='__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])
    main(weekcount,ticknum,winnum)