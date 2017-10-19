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
from keras.layers.convolutional_recurrent import ConvLSTM2D
from SampConv import SamplingConvLSTM2D
from keras import backend as K
from sklearn.metrics import log_loss, f1_score
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
    conv1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(att)
    bn1 = BatchNormalization()(conv1)
    conv2 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=False)(bn1)
    bn2 = BatchNormalization()(conv2)

    # Decoder here

    rep0 = Flatten()(bn2)
    rep1 = RepeatVector(labseq)(rep0)
    rep2 = Reshape((labseq,5,seqsize-2,64))(rep1)
    dec1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(rep2)
    bn3 = BatchNormalization()(dec1)
    res = Reshape((labseq,5*(seqsize-3)*64))(bn3)

    out = [TimeDistributed(Dense(5, activation='softmax', name=('predict_out' + str(i))))(res) for i in range(5)]
    # Now we combine the confidence output and prediction output into a single output...
    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# Train model here
def main(weekcount,ticknum,winnum,large,restart):
    labseq = winnum
    stocks = 5
    classes = 5
    numtests=1 # silly placeholder

    mainadd = '/home/mharmon/FinanceProject/ModelResults/tickcorrectweek'+str(ticknum) + 'win' + str(winnum)
    modeladd = mainadd + '/Models/tickmodel'
    batchsize=256
    seqsize = ticknum
    model = build_rnn(seqsize,batchsize,labseq)
    month_len = int(8064)
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...
    old_numtests = 0
    for overall in range(large,4):
        address  = '/home/mharmon/FinanceProject/Data/tickdata/traincorrectweek'+str(ticknum) + 'cnn'+str(overall)+'.pkl'

        # Start Here
        if overall==0 and restart==0:
            data,labels,dates = pkl.load(open(address,'rb'))
            data = np.swapaxes(data,2,4)
            data = np.swapaxes(data,2,3)
            first_length = (len(data) - 8064)/2016
            numtests = int(first_length)

        # Continuation (nothing breaks..)
        if overall>0 and restart==0 and weekcount ==0:
            # Square away everything with the data
            print('Continue with data from')
            print(address)
            newdata,newlabels,newdates = pkl.load(open(address,'rb'))
            newdata = np.swapaxes(newdata,2,4)
            newdata = np.swapaxes(newdata,2,3)

            old_data = data[int(numtests*week):]
            old_labels = labels[int(numtests*week):]
            old_dates = dates[int(numtests*week):]

            print('Old data Shape Added')
            print(old_data)
            print('New data is shape')
            print(newdata.shape)
            print()
            data = np.vstack((old_data,newdata))
            labels = np.vstack((old_labels,newlabels))
            dates = np.concatenate((old_dates,newdates))

            numtests = int((len(data)-month_len)/float(week))




        # Restart, no sector overlap

        if overall ==0 and restart ==1:
            data,labels,dates = pkl.load(open(address,'rb'))
            data = np.swapaxes(data,2,4)
            data = np.swapaxes(data,2,3)
            numtests = int((len(data)-month_len)/float(week))

        if overall>0 and restart==1 and weekcount>=4:

            newdata,newlabels,newdates = pkl.load(open(address,'rb'))
            newdata = np.swapaxes(newdata,2,4)
            newdata = np.swapaxes(newdata,2,3)
            print('New Data is from')
            print(address)
            old_add = '/home/mharmon/FinanceProject/Data/tickdata/traincorrectweek'+str(ticknum) + 'cnn'+str(overall-1)+'.pkl'
            old_data,old_labels,old_dates = pkl.load(open(old_add,'rb'))
            old_data = np.swapaxes(old_data,2,4)
            old_data = np.swapaxes(old_data,2,3)

            numtests = int((len(old_data)-month_len)/float(week))
            old_data = old_data[int(numtests*week):]
            old_labels = old_data[int(numtests*week):]
            old_dates = old_data[int(numtests*week):]
            print('Old data is from')
            print(old_add)
            print()
            print(old_data.shape)
            print(newdata.shape)
            data = np.vstack((old_data,newdata))
            labels = np.vstack((old_labels,newlabels))
            dates = np.concatenate((old_dates,newdates))

            numtests = int((len(data)-month_len)/float(week))

        # Need to to a check if it broke in between data, which is super incovenient, but possible

        if weekcount<=4 and restart==1 and overall>0:
            print('Old data is from')
            address  = '/home/mharmon/FinanceProject/Data/tickdata/traincorrectweek'+str(ticknum) + 'cnn'+str(overall-1)+'.pkl'
            print(address)
            old_data,old_labels,old_dates = pkl.load(open(address,'rb'))
            old_data = np.swapaxes(old_data,2,4)
            old_data = np.swapaxes(old_data,2,3)
            address  = '/home/mharmon/FinanceProject/Data/tickdata/traincorrectweek'+str(ticknum) + 'cnn'+str(overall)+'.pkl'
            print('New Data is from')
            print(address)
            print()
            newdata,newlab,newdates = pkl.load(open(address,'rb'))
            newdata = np.swapaxes(newdata,2,4)
            newdata = np.swapaxes(newdata,2,3)

            num_weeks = int(len(old_data)/week)

            old_data = old_data[int(num_weeks*week-week*(4-weekcount)):]
            old_labels = old_labels[int(num_weeks*week-week*(4-weekcount)):]
            old_dates = old_dates[int(num_weeks*week-week*(4-weekcount)):]

            print(old_data.shape)
            print(newdata.shape)
            data = np.vstack((old_data,newdata))
            labels = np.vstack((old_labels,newlab))
            dates = np.concatenate((old_dates,newdates))

            numtests = int((len(data)-month_len)/float(week))


        pred_runs=1

        beg = 0+week*weekcount
        end = month_len+week*weekcount

        f1testavg = np.zeros((int(numtests),winnum))
        f1store = np.zeros((numtests, 5, winnum, 5))
        f1wherestore=[]


        # These two cases actually take care of everything
        if weekcount>0:
            print('Loading Model ')
            modelsavepath = mainadd + '/Models/tickmodel'+str(weekcount-1) + 'sector' + str(overall) + '.h5'
            print(modelsavepath)
            print()
            model.load_weights(modelsavepath)

            # Load the old f1 scores...
            f1save = mainadd+'/f1score' + 'sector'+str(overall)+'.npy'
            f1avgsave = mainadd + '/f1avg'+'sector'+str(overall)+'.npy'
            fwheresave = mainadd + '/fwhere'+'sector'+str(overall)+'.pkl'
            f1store = np.load(f1save)
            f1testavg = np.load(f1avgsave)
            f1wherestore = pkl.load(open(fwheresave, 'rb'))

            if len(f1store)<numtests:
                for z in range(numtests-len(f1store)):
                    f1store=np.append(f1store,np.zeros((1,5,winnum,5)))
                    f1testavg=np.append(f1testavg,np.zeros((1,winnum)))

            f1testavg = np.reshape(f1testavg,(numtests,1))
            f1store = np.reshape(f1store,(numtests,5,winnum,5))


        if weekcount==0 and overall >0:
            modelsavepath = mainadd + '/Models/tickmodel' + str(old_numtests-1) + 'sector'+str(overall-1)+'.h5'

            print('Loading Model')
            print(modelsavepath)
            print()
            model.load_weights(modelsavepath)

        for i in range(weekcount,numtests):
            modelsavepath = mainadd + '/Models/tickmodel' + str(i) + 'sector'+str(overall)+ '.h5'
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
                    cross_entropy_check = np.zeros((stocks,winnum))
                    if overall==0 and weekcount==0:
                        print(valpred[0].shape)
                    for ti in range(winnum):
                        for f in range(stocks):
                            tempvalf1 = f1_score(y_true=np.argmax(valy[f][:,ti,:], axis=1),
                                                           y_pred=np.argmax(valpred[f][:,ti,:], axis=1), average=None)
                            current_f1[ti]+= np.average(tempvalf1)/(stocks)

                            cross_entropy_check[f,ti] = log_loss(valy[f][:,ti,:],valpred[f][:,ti,:])



                    #current_val = hist.history['val_loss'][0]
                    print('')
                    print('Window ' + str(i))
                    print('Round ' + str(j))
                    print('Sector ' + str(overall))
                    print('Epoch Took %.3f Seconds' % (endtime - firsttime))
                    print('Train Loss is ' + str(hist.history['loss'][0]))
                    print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                    print('F1 Score is %.3f' %np.mean(current_f1))
                    print('Checking Cross Entropy Values (Time Goes Right ---->')
                    print(cross_entropy_check)
                    
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
                tempred = np.zeros((week,5,winnum,5),'float32')
                for p in range(5):
                    tempred[:,p,:,:] =predictions[p]


                adds=mainadd + '/Predictions/predictions'+str(i)+'run'+str(r)+'sector'+str(overall)+'.pkl'
                pkl.dump([tempred,np.array(test_dates)],open(adds,'wb'))
            # This will need to be done for each softmax function
            # This gathers all of the f1 scores in a reasonable manner...

            for ti in range(winnum):
                for f in range(stocks):
                    f1tempscore = f1_score(y_true=np.argmax(testy[f][:,ti,:], axis=1),
                                                   y_pred=np.argmax(predictions[f][:,ti,:], axis=1), average=None)
                    f1where = np.unique(np.argmax(testy[f][:,ti,:], axis=1))
                    f1wherestore += [f1where.tolist()]
                    f1testavg[i,ti] += np.average(f1tempscore)/(stocks*winnum)

                    for c in range(len(f1where)):
                        f1store[i, f, ti,f1where[c]] = f1tempscore[c]

            f1save = mainadd + '/f1score'+'sector'+str(overall)+'.npy'
            f1avgsave = mainadd + '/f1avg'+'sector'+str(overall)+'.npy'
            fwheresave = mainadd + '/fwhere'+'sector'+str(overall)+'.pkl'
            np.save(f1save, np.array(f1store))
            np.save(f1avgsave,f1testavg)
            pkl.dump(f1wherestore,open(fwheresave,'wb'))
            # I should only have 25 histograms for all test runs
            print()
            print('Finished Window ' + str(i))
            print('Training Loss is ' + '%.5f' % hist.history['loss'][0])
            print('Validation Loss is ' + '%.5f' % hist.history['val_loss'][0])
            print('Best F1 is %.5f' %best_f1)


        weekcount=0
        restart = 0
        old_numtests=numtests
    return

if __name__=='__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])
    large = int(sys.argv[4])
    restart = int(sys.argv[5])
    main(weekcount,ticknum,winnum,large,restart)