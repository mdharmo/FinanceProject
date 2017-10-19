# Author: Mark Harmon
# Purpose: Create autoencoder that can help deduce the current state of the market and identify changes within
# the market
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation,LSTM,Reshape,RepeatVector,Permute,Flatten,Input,Dropout
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.merge import Multiply
from keras.layers.wrappers import TimeDistributed
import pickle as pkl
import numpy as np
import sys
import keras
from keras.models import Model,load_model, Sequential
from keras.layers.normalization import BatchNormalization
import time
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,History

def build_ae(seqsize,stocks):
    # Make my autoencoder here
    # Pretty sure that I need to use the model api to make this work...

    main_input = Input(shape=(4,stocks,seqsize,1), dtype='float32', name='main_input')
    # This is where we do attention (This may need to change, attention for conv models seems odd).
    '''
    a1 = Reshape((4,stocks*seqsize))(new_main)
    a2 = TimeDistributed(Dense(1, activation='tanh'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(stocks*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,stocks,seqsize,1))(a6)
    att = Multiply()([new_main,a7])
    '''
    # Encoder first
    conv1 = ConvLSTM2D(filters=32,kernel_size=(1,3),return_sequences=False,padding='same')(main_input)
    bn1 = BatchNormalization()(conv1)

    bottle_neck=64
    btlenk = Conv2D(filters=bottle_neck,kernel_size=(1,1),padding='valid',kernel_regularizer=keras.regularizers.l1(0.1))(bn1)
    bn2 = BatchNormalization()(btlenk)

    # Now upsample back up
    #us = UpSampling2D(size=(1,10))(bn2)

    bef = Flatten()(bn2)
    rep = RepeatVector(4)(bef)
    rep2 = Reshape((4,stocks,seqsize,64))(rep)

    conv2 = ConvLSTM2D(32,return_sequences=True,kernel_size=(1,3),padding='same')(rep2)
    bn3 = BatchNormalization()(conv2)

    out = ConvLSTM2D(filters=1,kernel_size=(1,1),return_sequences=True,activation='linear')(bn3)

    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='mean_squared_error')

    return model

def build_reg_ae(inshape):

    main_input = Input(shape=(880),dtype='float32',name='main_input')

    #d1 = Dense(100)(main_input)
    d2 = Dense(1,activation='relu',kernel_regularizer=keras.regularizers.l2)(main_input)
    #d3 = Dense(100,activation='relu')(d2)
    out = Dense(22*10*4,activation='sigmoid',kernel_regularizer=keras.regularizers.l2)(d2)
    model = Model(main_input,out)

    model.compile(optimizer=SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=True), loss='mse')



    return model

def seq_model(inshape):
    model = Sequential()
    model.add(Dense(200,input_dim=880,activation='relu'))
    model.add(Dense(100, activation='relu',kernel_regularizer=keras.regularizers.l1(0.001)))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(880,activation='linear'))
    model.compile(optimizer='sgd',loss='mse')

    return model
def main(weekcount,ticknum,winnum):
    stocks = 22
    avg_store = np.zeros((30, 2))
    mainadd = '/home/mharmon/FinanceProject/ModelResults/aeweek'
    address  = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday'+str(ticknum) + 'win' + str(winnum)+'sector0.pkl'
    batchsize=64
    data,labels,dates = pkl.load(open(address,'rb'))
    print(np.max(data))
    month_len = int(8064)
    real_two_month = int(8064*2)
    data = np.swapaxes(data,2,4)
    data = np.swapaxes(data,2,3)

    #data = data[:,:,0,:]
    # Need to additionally fit the smotes for training purposes...
    np.random.seed(1)
    num_tests = 1
    #data = np.reshape(data,(len(data),22*10*4))
    #print(data.shape)
    #print(np.unique(data))
    model = build_ae(ticknum,stocks)
    val = 36 +1
    for k in range(num_tests):

        #beg = int(k*real_two_month)
        #end = int((k*real_two_month+month_len))

        beg = 288
        end = val*288

        # Load model if one is availabl


        modelsavepath = mainadd + '/Models/tickmodel' + str(0) + '.h5'

        epochx = data[beg:end]

        vallen = int(len(epochx) * 0.15)

        trainx = epochx[0:len(epochx) - vallen]
        valx = epochx[len(epochx) - vallen:]

        best_val = 100
        patience = 0
        epochs=0
        #while patience<5 and epochs<50:
        firsttime = time.time()
        rtrainx = np.copy(trainx) + 0.25*np.random.normal(0,1.0,size=trainx.shape)
        rtrainx = np.clip(rtrainx,0,1.)

        #rvalx = np.copy(valx) + 0.25*np.random.normal(0,1.0,size=valx.shape)
        #rvalx = np.clip(rvalx,0,1.)

        es = EarlyStopping(patience=3)
        es2 = EarlyStopping(monitor='loss')
        hist = model.fit(trainx, trainx, batch_size=batchsize,verbose=1, epochs=100,validation_data=(valx,valx),callbacks=[es,es2])
        endtime = time.time()

        # Pretraining done, noow train on that singular day...
        epochx = data[end:end+288]
        vallen = int(len(epochx)*0.15)
        trainx = epochx[0:len(epochx) - vallen]
        valx = epochx[len(epochx) - vallen:]


        hist = model.fit(trainx, trainx, batch_size=1,verbose=1, epochs=25,validation_data=(valx,valx),callbacks=[es,es2])

        '''
        current_val = hist.history['val_loss'][0]
        print('')
        print('Window ' + str(0))
        print('Epoch Took %.3f Seconds' % (endtime - firsttime))
        print('Train Loss is ' + str(hist.history['loss'][0]))
        print('Validation Loss is ' + str(hist.history['val_loss'][0]))
        epochs+=1
        if np.mean(current_val)<best_val:
            best_val = np.mean(current_val)
            patience = 0
            model.save(modelsavepath)
            print('New Saved Model')

            # Save model
        else:
            patience+=1

        '''
        #del model
        #model = load_model(modelsavepath)

        #print(model.layers[-1].get_weights()[0])

        #first_half_data = np.concatenate((rtrainx,rvalx),axis=0)
        hour_len = 12
        my_range = int((288)/12.)
        validation_storage = []
        first = val*288
        last = first+hour_len
        x_first = np.arange(first,int(first+288),12)
        first_avg=0
        total_random = np.random.normal(0,1.0,size=(real_two_month,880))
        count=0
        for i in range(my_range):
            #rtestx = first_half_data[count:count+12]
            testx = data[first:last]
            hist = model.evaluate(testx,testx, verbose=0)
            validation_storage.append(hist)
            first_avg += hist/my_range
            count+=12

            # I should only have 25 histograms for all test runs
            print()
            print('Finished Window ' + str(i))
            print('Validation Loss is ' + '%.5f' % hist)

            first+=hour_len
            last+=hour_len


        first = val*288 + 288
        last = first+hour_len
        x_next = np.arange(first,int(first+288),12)
        second_avg = 0
        for i in range(len(x_next)):
            testx = data[first:last]
            #rtestx = np.copy(testx) + 0.25*total_random[count:count+12]
            #rtestx = np.clip(rtestx,0,1.)
            hist = model.evaluate(testx,testx, verbose=0)
            validation_storage.append(hist)
            second_avg+=hist/my_range
            count+=12


            # I should only have 25 histograms for all test runs
            print()
            print('Finished Window ' + str(i))
            print('Validation Loss is ' + '%.5f' % hist)

            first+=hour_len
            last+=hour_len

        avg_store[k,0]=first_avg
        avg_store[k,1]=second_avg

        x_total = np.concatenate((x_first,x_next))
        fig_save = mainadd + '/Figures/PredictionZone' + str(k) +'.png'
        plt.figure(k)
        plt.plot(x_total,validation_storage,linewidth=2)
        plt.title('Error by Hour (Last Point is ' + str(last)+')')
        plt.xlabel('Hour')
        plt.ylabel('Loss')
        plt.savefig(fig_save)
        plt.close()



        # Do another test

    npsave = mainadd+'/differences.pkl'
    pkl.dump(avg_store,open(npsave,'wb'))
    return

if __name__ == '__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])

    main(weekcount,ticknum,winnum)