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
from sklearn.metrics import log_loss

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
def main(weekcount,ticknum,winnum):
    labseq = winnum
    stocks = 5
    numtests=100 # number test periods/epochs

    data_add = '/home/mharmon/FinanceProject/Data/tickdata/randomtrain10cnn4.pkl'
    data,labels = pkl.load(open(data_add,'rb'))
    data = np.swapaxes(data,2,4)
    data = np.swapaxes(data,2,3)
    mainadd = '/home/mharmon/FinanceProject/ModelResults/test'+str(ticknum) + 'win' + str(winnum)
    batchsize=64
    seqsize = ticknum
    model = build_rnn(seqsize,batchsize,labseq)
    month_len = int(8064)
    week = int((month_len)/4.)


    pred_runs=1
    f1testavg = np.zeros((int(numtests),winnum))
    modelsavepath = mainadd + '/Models/random' + str(np.max((0,weekcount-1))) +'.h5'

    beg = 0+week*weekcount
    end = month_len+week*weekcount
    f1store = np.zeros((numtests, 5, winnum, 5))
    f1wherestore=[]
    totalpred = []


    # Now going to actually make the labels...

    # Load model if one is available

    if weekcount>0:
        model.load_weights(modelsavepath)

        # Load the old f1 scores...
        f1save = mainadd+'/f1score' + '.npy'
        f1avgsave = mainadd + '/f1avg'+'.npy'
        fwheresave = mainadd + '/fwhere'+'.pkl'
        f1store = np.load(f1save)
        f1testavg = np.load(f1avgsave)
        f1wherestore = pkl.load(open(fwheresave, 'rb'))


    for i in range(weekcount,numtests):
        modelsavepath = mainadd + '/Models/tickmodel' + str(i) + '.h5'
        epochx = data[beg:end]
        epoch_lab = labels[beg:end]
        testx = data[end:end+week]
        test_lab = labels[end:end+week]
        testy = [test_lab[:,:,i,:] for i in range(len(test_lab[0,0,:,0]))]
        # Do at least 5 validation sets here..

        vallen = int(len(epochx) * 0.15)
        rng_vec = np.random.permutation(len(epochx))

        trainx = epochx[rng_vec[0:len(epochx) - vallen]]
        trainy = [epoch_lab[rng_vec[0:len(epochx) - vallen],:, i, :] for i in range(len(epoch_lab[0,0, :, 0]))]

        valx = epochx[rng_vec[len(epochx) - vallen:]]
        valy = [epoch_lab[rng_vec[len(epochx) - vallen:],:, i, :] for i in range(len(epoch_lab[0, 0,:, 0]))]

        print(trainx.shape)
        print(trainy[0].shape)
        print(valx.shape)
        print(valy[0].shape)
        print(vallen)
        for j in range(1):

            #trainx, trainy = smote_conv_seq(trainx, trainy,seqsize,winnum,stocks,classes)

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
                for ti in range(winnum):
                    for f in range(stocks):
                        tempvalf1 = metrics.f1_score(y_true=np.argmax(valy[f][:,ti,:], axis=1),
                                                       y_pred=np.argmax(valpred[f][:,ti,:], axis=1), average=None)
                        current_f1[ti]+= np.average(tempvalf1)/(stocks)

                        cross_entropy_check[f,ti] = log_loss(valy[f][:,ti,:],
                                                                                valpred[f][:,ti,:])


                print('')
                print('Window ' + str(i))
                print('Round ' + str(j))
                print('Epoch Took %.3f Seconds' % (endtime - firsttime))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                print('F1 Score is %.3f' %np.mean(current_f1))
                print('Checking Cross Entropy Values (Time Goes Right ---->)')
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


            adds=mainadd + '/Predictions/predictions'+str(i)+'run'+str(r)+'.pkl'
            pkl.dump([tempred,testy],open(adds,'wb'))
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

        f1save = mainadd + '/f1score'+'.npy'
        f1avgsave = mainadd + '/f1avg'+'.npy'
        fwheresave = mainadd + '/fwhere'+'.pkl'
        np.save(f1save, np.array(f1store))
        np.save(f1avgsave,f1testavg)
        pkl.dump(f1wherestore,open(fwheresave,'wb'))
        # I should only have 25 histograms for all test runs
        print()
        print('Finished Window ' + str(i))
        print('Training Loss is ' + '%.5f' % hist.history['loss'][0])
        print('Validation Loss is ' + '%.5f' % hist.history['val_loss'][0])
        print('Best F1 is %.5f' %best_f1)


    return

if __name__=='__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])
    main(weekcount,ticknum,winnum)