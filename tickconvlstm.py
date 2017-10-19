# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense,Flatten
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import pickle as pkl
import numpy as np
from sklearn import metrics
import time
from imblearn.over_sampling import SMOTE as smote
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D

def build_rnn(seqsize):
    main_input = Input(shape=(4,5,10,1), dtype='float32', name='main_input')
    conv1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(main_input)
    bn1 = BatchNormalization()(conv1)
    conv2 = ConvLSTM2D(filters=128,kernel_size=(1,2),return_sequences=True)(bn1)
    bn2 = BatchNormalization()(conv2)
    r1 = Reshape((4,5*4*128))(bn2)
    gru1 = GRU(512)(r1)
    bn3 = BatchNormalization()(gru1)

    # I'm looping right here, but this may be overly complicated
    out = [Dense(5, activation='softmax', name=('main_output' + str(i)))(bn3) for i in range(5)]
    model = Model(main_input, out)

    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# Train model here
def main(weekcount):
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train10cnn0.pkl'
    mydates = []
    data,labels,dates = pkl.load(open(address,'rb'))
    seqsize = 10
    batchsize=256
    seq_num = 4
    data = (data - np.mean(data,axis=0))/np.std(data,axis=0)
    model = build_rnn(seqsize)
    month_len = 8064
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...


    numtests = 30
    f1testavg = np.zeros(numtests)
    modelsavepath = '/home/mharmon/FinanceProject/ModelResults/conv1d/models/tickmodel'+str(weekcount-1)+'.h5'


    beg = 0+weekcount*week
    end = month_len+weekcount*week
    pngcount=0
    f1store = np.zeros((numtests, 5, 5))
    stock1 = [[], [], [], [], []]
    stock2 = [[], [], [], [], []]
    stock3 = [[], [], [], [], []]
    stock4 = [[], [], [], [], []]
    stock5 = [[], [], [], [], []]
    f1wherestore=[]
    totalpred = []


    # Load model if one is availabl

    if weekcount>0:
        model = load_model(modelsavepath)

        # Load the old f1 scores...
        f1save = '/home/mharmon/FinanceProject/ModelResults/conv1d/f1score.npy'
        f1avgsave = '/home/mharmon/FinanceProject/ModelResults/conv1d/f1avg.npy'
        fwheresave = '/home/mharmon/FinanceProject/ModelResults/conv1d/fwhere.pkl'
        f1store = np.load(f1save)
        f1testavg = np.load(f1avgsave)
        f1wherestore = pkl.load(open(fwheresave, 'rb'))

    for i in range(weekcount, numtests):
        modelsavepath = '/home/mharmon/FinanceProject/ModelResults/conv1d/models/tickmodel' + str(i) + '.h5'
        epochx = data[beg:end]
        epoch_lab = labels[beg:end]
        testx = data[end:end + week]
        test_lab = labels[end:end + week]
        test_dates = dates[end:end + week]
        testy = [test_lab[:, i, :] for i in range(len(test_lab[0, :, 0]))]
        # Do at least 5 validation sets here..

        vallen = int(6*batchsize)

        trainx = epochx[0:len(epochx) - vallen]
        trainy = [epoch_lab[0:len(epochx) - vallen, i, :] for i in range(len(epoch_lab[0, :, 0]))]

        valx = epochx[len(epochx) - vallen:]
        valy = [epoch_lab[len(epochx) - vallen:, i, :] for i in range(len(epoch_lab[0, :, 0]))]

        for j in range(2):

            trainx, trainy = sampling_from_sme(trainx, trainy, seqsize,seq_num)

            trainx = np.swapaxes(trainx,2,4)
            trainx = np.swapaxes(trainx,2,3)

            valx = np.swapaxes(valx,2,4)
            valx = np.swapaxes(valx,2,3)

            best_f1 = -1
            patience = 0
            while patience < 5:
                firsttime = time.time()
                hist = model.fit(trainx, trainy, batch_size=batchsize, verbose=0, epochs=1,
                                 validation_data=(valx, valy))
                endtime = time.time()

                valpred = model.predict(valx, verbose=0)
                # Here is where I calculate f1 score to determine when to save my model...
                current_f1 = 0
                for f in range(5):
                    tempvalf1 = metrics.f1_score(y_true=np.argmax(valy[f], axis=1),
                                                 y_pred=np.argmax(valpred[f], axis=1), average=None)
                    current_f1 += np.average(tempvalf1) / 5.

                # current_val = hist.history['val_loss'][0]
                print('')
                print('Window ' + str(i))
                print('Round ' + str(j))
                print('Epoch Took %.3f Seconds' % (endtime - firsttime))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                print('F1 Score is %.3f' % current_f1)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    patience = 0
                    model.save(modelsavepath)
                    print('New Saved Model')

                    # Save model
                else:
                    patience += 1

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

        testx = np.swapaxes(testx, 2, 4)
        testx = np.swapaxes(testx, 2, 3)

        predictions = model.predict(testx, batch_size=32, verbose=1)
        tempred = np.zeros((week, 5, 5))
        for p in range(5):
            tempred[:, p, :] = predictions[p]

        adds = '/home/mharmon/FinanceProject/ModelResults/conv1d/Predictions/predictions' + str(i) + '.pkl'
        pkl.dump([tempred, np.array(test_dates)], open(adds, 'wb'))
        # This will need to be done for each softmax function
        # This gathers all of the f1 scores in a reasonable manner...
        for f in range(5):
            f1tempscore = metrics.f1_score(y_true=np.argmax(testy[f], axis=1),
                                           y_pred=np.argmax(predictions[f], axis=1), average=None)
            f1where = np.unique(np.argmax(testy[f], axis=1))
            f1wherestore += [f1where.tolist()]
            f1testavg[i] += np.average(f1tempscore) / 5.

            for c in range(len(f1where)):
                f1store[i, f, f1where[c]] = f1tempscore[c]

        f1save = '/home/mharmon/FinanceProject/ModelResults/conv1d/f1score.npy'
        f1avgsave = '/home/mharmon/FinanceProject/ModelResults/conv1d/f1avg.npy'
        fwheresave = '/home/mharmon/FinanceProject/ModelResults/conv1d/fwhere.pkl'
        np.save(f1save, np.array(f1store))
        np.save(f1avgsave, f1testavg)
        pkl.dump(f1wherestore, open(fwheresave, 'wb'))
        # I should only have 25 histograms for all test runs
        print()
        print('Finished Window ' + str(i))
        print('Training Loss is ' + '%.5f' % hist.history['loss'][0])
        print('Validation Loss is ' + '%.5f' % hist.history['val_loss'][0])
        print('Best F1 is %.5f' % best_f1)

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
        figmain = '/home/mharmon/FinanceProject/ModelResults/conv1d/Figures/HistogramAllWeeksStock'

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

if __name__ == '__main__':
    weekcount = 0
    main(weekcount)