# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function

from keras.layers import Input, Embedding, GRU, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
import time

def build_rnn():
    main_input = Input(shape=(10,300,), dtype='float32', name='main_input')
    gru1 = GRU(512,return_sequences=True)(main_input)
    gru2 = GRU(512)(gru1)
    gru3 = GRU(512)(gru2)
    # I'm looping right here, but this may be overly complicated
    out = [Dense(5, activation='softmax', name=('main_output' + str(i)))(gru3) for i in range(25)]
    model = Model(main_input, out)

    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def manipulate_data(data,labels):
    # This is where I'm going to create my batches.  I need to reshape the data into the proper
    # format and proper labels.  This code needs to be extremely efficient. This is needed
    # for the walk forward methodology.  I will deal with this later

    data = np.array(np.reshape(data,(int(len(data)/10.),10,300)))

    # I need to remember that the labels need to be a list

    ytrain = [labels[:, i, :] for i in range(len(labels[0,:,0]))]
    return data,ytrain

# Train model here
def main(address,epoch,batchsize):
    onlyfiles = [f for f in listdir(address) if isfile(join(address, f))]
    model = build_rnn()

    modelsavepath = '/home/mharmon/FinanceProject/ModelResults/spread/spreadmodel.hdf5'
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    modelsave = ModelCheckpoint(modelsavepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False, mode='auto', period=1)

    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.
    #model.train_on_batch(data, labels, epochs=1, batches=100)

    # Set up for the walkforward method
    count = 0
    begloop = 0
    fulladdress = address + onlyfiles[0]
    datanew, labelsnew = pkl.load(open(fulladdress, 'rb'))
    datanew, labelsnew = manipulate_data(datanew, labelsnew)
    window = 0
    while len(onlyfiles)>0:

        start = time.time()
        stepforward = 10000
        windowprediction = 5000
        datasize = 30000


        # First seperate out the test set completely from the rest of the data
        if begloop == 0:
            totaldata = datanew
            totallabels = labelsnew
            begloop+=1
        else:

            # See if we have been through all of dataold yet
            if stepforward*count >= len(dataold):
                del dataold
                del labelsold
                dataold = datanew
                labelsold = labelsnew
                fulladdress = address + onlyfiles[0]
                datanew,labelsnew = pkl.load(open(fulladdress,'rb'))
                datanew, labelsnew = manipulate_data(datanew, labelsnew)
                del onlyfiles[0]
                count = 0

            firstlength = len(dataold[stepforward*count,:])

            need = datasize - firstlength

            # If the second dataset does not accomadate the need
            if need-len(datanew)<0:
                dataold = np.concatenate((dataold,datanew))
                labelsold = np.concatenate((labelsold,labelsnew))
                firstlength = len(dataold[stepforward*count,:])
                need = datasize-firstlength
                fulladdress = address + onlyfiles[0]
                datanew,labelsnew = pkl.load(open(fulladdress,'rb'))
                datanew, labelsnew = manipulate_data(datanew, labelsnew)
                del onlyfiles[0]



            totaldata = np.concatenate(dataold[stepforward*count,:],datanew[0:need])
            totallabels = np.concatenate(labelsold[stepforward*count,:],datanew[0:need])



        # First take our our test set
        testx = totaldata[-windowprediction:]
        testy = [stuff[-windowprediction:] for stuff in totallabels]

        availdata = totaldata[0:-windowprediction]
        availlabel = [stuff[0:-windowprediction] for stuff in totallabels]


        # Validation split
        valpiece = len(availdata) - int(0.3 * len(availdata))
        vec = np.arange(len(availdata))
        vec = np.random.permutation(vec)
        valx = availdata[vec[valpiece:]]
        valy = [stuff[vec[valpiece:]] for stuff in availlabel]

        # Training Split
        trainx = availdata[vec[0:valpiece]]
        trainy = [stuff[vec[0:valpiece]] for stuff in availlabel]

        oldloss = 100
        currentloss = 50
        round = 0
        while abs(oldloss-currentloss)>1 or round<5:
            oldloss = currentloss
            hist = model.fit(trainx,trainy,batch_size=100,verbose = 0,epochs=epoch,validation_data=[valx,valy],callbacks = [es,modelsave])
            currentloss = np.mean(hist.history['val_loss'])
            round+=1

            # Mix up the data again for the next round
            vec = np.random.permutation(vec)
            valx = availdata[vec[valpiece:]]
            valy = [stuff[vec[valpiece:]] for stuff in availlabel]

            # Training Split
            trainx = availdata[vec[0:valpiece]]
            trainy = [stuff[vec[0:valpiece]] for stuff in availlabel]

        end = time.time()
        totaltime = end-start

        dataold = datanew
        labelsold = labelsnew

        testloss = model.evaluate(testx, testy, verbose=0, sample_weight=None)
        testclasses = model.predict_classes(testx, batch_size=32, verbose=1)

        print()
        print('Finished Window ' + str(window))
        print(str(epoch) + ' Epochs Took ' + str(totaltime) + ' Seconds')
        print('Training Loss is ' +'%.5f' %hist.history['loss'][0])
        print('Validation Loss is ' + '%.5f' %hist.history['val_loss'][0])

        acclen = int(len(totaldata[0,0,:])/12)
        accstore = np.zeros(acclen)
        for a in range(int(len(totaldata[0,0,:])/12)):
            accstring = 'val_main_output'+str(a)+'_acc'
            accstore[a] = hist.history[accstring][0]

        print('Validation Accuracy is ' + '%.5f' %(np.mean(accstore)*100))
        window +=1


    return hist



