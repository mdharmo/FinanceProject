# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense
from keras.models import Model,load_model,Sequential
from keras.layers.normalization import BatchNormalization
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import metrics
import time
from imblearn.over_sampling import SMOTE as smote
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K

def build_rnn(seqsize):

    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(GRU(128, input_shape=(20,5,)))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(3):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    return rnn

def sampling_from_sme(data,lab,seqsize,stocks):

    #stocks = 5
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
                            fitdat = np.reshape(data[tempvec],(len(data[tempvec]),seqsize*stocks))
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


    totaldata = np.reshape(totaldata,(len(totaldata),seqsize,stocks))
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
def main():
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train10.pkl'

    data,labels = pkl.load(open(address,'rb'))
    seqsize = 10
    data = (data - np.mean(data,axis=0))/np.std(data,axis=0)
    ae = build_ae(seqsize)
    rnn = build_rnn(seqsize)
    month_len = 8064
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...

    aesave = '/home/mharmon/FinanceProject/ModelResults/tickae/aemodel.hdf5'

    mc = ModelCheckpoint(aesave, monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='auto', period=1)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    numtests = 30
    f1testavg = np.zeros(numtests)
    modelsavepath = '/home/mharmon/FinanceProject/ModelResults/tickae/tickmodel.hdf5'

    beg = 0
    end = month_len
    pngcount=0
    f1store = np.zeros((numtests, 5, 5))
    stock1 = [[], [], [], [], []]
    stock2 = [[], [], [], [], []]
    stock3 = [[], [], [], [], []]
    stock4 = [[], [], [], [], []]
    stock5 = [[], [], [], [], []]
    f1wherestore=[]
    for i in range(numtests):

        epochx = data[beg:end]
        epoch_lab = labels[beg:end]
        testx = data[end:end+week]
        test_lab = labels[end:end+week]
        testy = [test_lab[:,i,:] for i in range(len(test_lab[0,:,0]))]
        # Do at least 5 validation sets here..

        vallen = int(len(epochx) * 0.2)
        trainx = epochx[0:len(epochx) - vallen]
        trainy = [epoch_lab[0:len(epochx) - vallen, i, :] for i in range(len(epoch_lab[0, :, 0]))]

        valx = epochx[len(epochx) - vallen:]
        valy = [epoch_lab[len(epochx) - vallen:, i, :] for i in range(len(epoch_lab[0, :, 0]))]

        for j in range(2):
            # Autoencoder training should be done right here
            aetrainx = np.reshape(trainx,(len(trainx),len(trainx[0,:,0])*len(trainx[0,0,:])))
            aevalx = np.reshape(valx,(len(valx),len(valx[0,:,0])*len(valx[0,0,:])))
            ae.fit(aetrainx,aetrainx,batch_size=64,epochs=200,validation_data=(aevalx,aevalx),callbacks=[mc,es])

            # Extract the values I need from the middle of the autoencoder

            ae.model.outputs = [ae.model.layers[3].output]
            newtrainx = ae.model.predict(aetrainx)
            newvalx = ae.model.predict(aevalx)
            newtrainx = np.reshape(newtrainx,(len(newtrainx),10,2))
            newvalx = np.reshape(newvalx,(len(newvalx),10,2))
            featsize = 2
            # Do smote on these values
            newtrainx, newtrainy = sampling_from_sme(newtrainx, trainy,seqsize,featsize)

            print(newtrainx.shape)
            best_f1 = -1
            patience = 0

            del ae
            ae = load_model(aesave)

            while patience<5:
                firsttime = time.time()
                hist = rnn.fit(newtrainx, newtrainy, batch_size=256, verbose=0, epochs=1, validation_data=(newvalx,valy))
                endtime = time.time()

                valpred = rnn.predict(newvalx,verbose=0)
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
                print('Epoch Took %.3f Seconds' % (endtime - firsttime))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                print('F1 Score is %.3f' %current_f1)
                if current_f1>best_f1:
                    best_f1 = current_f1
                    patience = 0
                    rnn.save(modelsavepath)
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

            del rnn
            rnn = load_model(modelsavepath)



        beg += week
        end += week

        aetestx = np.reshape(testx, (len(testx), len(testx[0, :, 0]) * len(testx[0, 0, :])))
        ae.model.outputs = [ae.model.layers[3].output]
        newtestx = ae.model.predict(aetestx)
        newtestx = np.reshape(newtestx,(len(newtestx),10,2))
        predictions = rnn.predict(newtestx, batch_size=32, verbose=1)

        del ae
        ae = load_model(aesave)

        # This will need to be done for each softmax function
        # This gathers all of the f1 scores in a reasonable manner
        for f in range(5):
            f1tempscore = metrics.f1_score(y_true=np.argmax(testy[f], axis=1),
                                           y_pred=np.argmax(predictions[f], axis=1), average=None)
            f1where = np.unique(np.argmax(testy[f], axis=1))
            f1wherestore += [f1where.tolist()]
            f1testavg[i]+= np.average(f1tempscore)/5.

            for c in range(len(f1where)):
                f1store[i, f, f1where[c]] = f1tempscore[c]

        f1save = '/home/mharmon/FinanceProject/ModelResults/tickae/f1score.npy'
        f1avgsave = '/home/mharmon/FinanceProject/ModelResults/tickae/f1avg.npy'
        fwheresave = '/home/mharmon/FinanceProject/ModelResults/tickae/fwhere.pkl'
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
        figmain = '/home/mharmon/FinanceProject/ModelResults/tickae/Figures/HistogramAllWeeksStock'

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
    main()