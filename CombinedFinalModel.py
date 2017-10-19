# Author: Mark Harmon
# Purpose: This is my first attempt at a total combine model.  It incorporates the following:
'''
1). Autoencoder to determine how far to go back for training data and when to stop classifying
2). Classifer for predictions
3). Sampling layers on the convolutional layers
*************** Note that I need to add the sampling methodology to dense layers as well...
'''
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation,LSTM,Reshape,RepeatVector,Permute,Flatten,Input
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.merge import Multiply
from keras.layers.wrappers import TimeDistributed
import pickle as pkl
import numpy as np
import sys
import keras
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import time
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.noise import GaussianNoise
from SampConv import SamplingConvLSTM2D
from sklearn import metrics

def build_ae(seqsize,stocks):
    # Make my autoencoder here
    # Pretty sure that I need to use the model api to make this work...

    main_input = Input(shape=(4,5,seqsize,1), dtype='float32', name='main_input')
    new_main = GaussianNoise(0.1)(main_input)
    # This is where we do attention (This may need to change, attention for conv models seems odd).
    a1 = Reshape((4,5*seqsize))(new_main)
    a2 = TimeDistributed(Dense(1, activation='tanh'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(5*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,5,seqsize,1))(a6)
    att = Multiply()([new_main,a7])

    # Encoder first
    conv1 = ConvLSTM2D(filters=16,kernel_size=(1,3),return_sequences=False,padding='same')(att)
    bn1 = BatchNormalization()(conv1)

    bottle_neck=1
    btlenk = Conv2D(filters=bottle_neck,kernel_size=(1,10),W_regularizer=keras.regularizers.l1(0.1))(bn1)
    bn2 = BatchNormalization()(btlenk)

    # Now upsample back up
    us = UpSampling2D(size=(1,10))(bn2)

    bef = Flatten()(us)
    rep = RepeatVector(4)(bef)
    rep2 = Reshape((4,5,seqsize,1))(rep)

    conv2 = ConvLSTM2D(16,return_sequences=True,kernel_size=(1,3),padding='same')(rep2)
    bn3 = BatchNormalization()(conv2)

    out = ConvLSTM2D(filters=1,kernel_size=(1,1),return_sequences=True,activation='linear')(bn3)

    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy'])

    return model

def build_classifier(seqsize,batchsize,labseq):

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
    att = Multiply()([main_input, a7])

    # Encoder first
    conv1 = SamplingConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(att)
    bn1 = BatchNormalization()(conv1)
    conv2 = SamplingConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=False)(bn1)
    bn2 = BatchNormalization()(conv2)

    # Decoder here

    rep0 = Flatten()(bn2)
    rep1 = RepeatVector(labseq)(rep0)
    rep2 = Reshape((labseq,5,seqsize-2,64))(rep1)
    dec1 = SamplingConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(rep2)
    bn3 = BatchNormalization()(dec1)
    res = Reshape((labseq,5*(seqsize-3)*64))(bn3)

    out = [TimeDistributed(Dense(5, activation='softmax', name=('predict_out' + str(i))))(res) for i in range(5)]
    # Now we combine the confidence output and prediction output into a single output...
    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def train_ae(data,labels,mainadd,f_ae):

    val_len = int(len(data)*0.15)
    rng_vec= np.random.permutation(range(len(data)))
    trainx = data[rng_vec[val_len:]]
    valx = data[0:val_len]

    modelsavepath = mainadd + '/Models/tickae0.h5'

    best_val = 100
    patience = 0
    while patience<5:
        firsttime = time.time()
        hist_ae = f_ae.fit(trainx_ae, trainx_ae, batch_size=batchsize,verbose=1, epochs=1,
                        validation_data=(valx_ae,valx_ae))
        endtime = time.time()

        current_val = hist_ae.history['val_loss'][0]
        print('')
        print('Window ' + str(0))
        print('Epoch Took %.3f Seconds' % (endtime - firsttime))
        print('Train Loss is ' + str(hist_ae.history['loss'][0]))
        print('Validation Loss is ' + str(hist_ae.history['val_loss'][0]))

        if np.mean(current_val)<best_val:
            best_val = np.mean(current_val)
            patience = 0
            f_ae.save_weights(modelsavepath)
            rho = best_val
            print('New Saved Model')

            # Save model
        else:
            patience+=1


    f_ae.load_weights(modelsavepath)

    return f_ae

def train_classifier():

    return

def main(weekcount,ticknum,winnum):
    stocks = 5
    mainadd = '/home/mharmon/FinanceProject/ModelResults/ae'+str(ticknum) + 'win' + str(winnum)
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train'+str(ticknum) + 'cnn0.pkl'
    batchsize=256
    data,labels,dates = pkl.load(open(address,'rb'))
    seqsize = ticknum
    f_ae = build_ae(seqsize,stocks)
    f_c = build_classifier(seqsize,batchsize,winnum)
    month_len = int(8064)
    week = int((month_len)/4.)
    data = np.swapaxes(data,2,4)
    data = np.swapaxes(data,2,3)

    # Smote is only needed to create the extra data for the classifier, though I haven't decided whether or not to
    # use it for smote-like purposes yet.  Diego did mention a special kind of smote, but I have not found code for it.
    # the algorithm seems somewhat involved.
    T_w = int(month_len/2.)
    # There are some interesting things I could do here.  I could just determine the drop-ff point for the autoencoder
    # for finding T_s based upon training twoo weeks.  This one parameter can define the second parameters, which would
    # fairly nice to have...


    # Stopping criterion values...
    sigma_mu = 0.1

    # How much to change the rho_hat,eta_hat, and mu_hat values
    alpha_rho_future=0.9
    alpha_rho_past = 0.9
    alpha_mu=0.9


    # Do first round here...
    beg_class = 0
    beg_ae = T_w
    end = month_len

    epochx_ae = data[beg_ae:end]
    epochx_c = data[beg_class:end]
    epochlab_c = labels[beg_class:end]
    rng_vec_ae = np.random.permutation(range(len(epochx_ae)))
    rng_vec_c = np.random.permutation(range(len(epochx_c)))

    val_len_ae =int(len(epochx_ae)*0.15)
    val_len_c = int(len(epochx_c)*0.15)

    trainx_ae = epochx_ae[rng_vec_ae[val_len_ae:]]
    trainx_c = epochx_c[rng_vec_c[val_len_c:]]
    trainy_c = [epochlab_c[rng_vec_c[val_len_c:],:, i, :] for i in range(len(epochlab_c[0,0, :, 0]))]

    valx_ae = epochx_ae[rng_vec_ae[0:val_len_ae]]
    valx_c = epochx_c[rng_vec_c[0:val_len_c]]
    valy_c = [epochlab_c[rng_vec_c[0:val_len_c],:, i, :] for i in range(len(epochlab_c[0, 0,:, 0]))]

    # Don't need this yet.

    '''
    testx_c = data[end:end+week]
    testlab_c = labels[end:end+week]
    test_dates = dates[end:end+week]
    testy_c = [testlab_c[:,:,i,:] for i in range(len(testlab_c[0,0,:,0]))]
    '''

    # First train the autoencoder to find the sigma_eta value
    # Also need to make sure that I do weight loading rather than full model deletion, because that breaks
    # the sampling convoluational layer.
    modelsavepath = mainadd + '/Models/tickae0.h5'

    best_val = 100
    patience = 0
    while patience<5:
        firsttime = time.time()
        hist_ae = f_ae.fit(trainx_ae, trainx_ae, batch_size=batchsize,verbose=1, epochs=1,
                        validation_data=(valx_ae,valx_ae))
        endtime = time.time()

        current_val = hist_ae.history['val_loss'][0]
        print('')
        print('Window ' + str(0))
        print('Epoch Took %.3f Seconds' % (endtime - firsttime))
        print('Train Loss is ' + str(hist_ae.history['loss'][0]))
        print('Validation Loss is ' + str(hist_ae.history['val_loss'][0]))

        if np.mean(current_val)<best_val:
            best_val = np.mean(current_val)
            patience = 0
            f_ae.save_weights(modelsavepath)
            rho = best_val
            print('New Saved Model')

            # Save model
        else:
            patience+=1

        del f_ae

    # Load the best autoencoder model
    f_ae.load_weights(modelsavepath)

    # Now do predictions on the previous times in order to find sigma_rho_past value...
    predx_ae = data[0:beg_ae]
    hist_pred_ae = f_ae.evaluate(predx_ae,predx_ae,verbose=0)
    rho_hat_past = hist_pred_ae['val_loss'][0]

    sigma_rho = np.abs(rho_hat_past-rho)

    # Now do the first training session of the classifier
    modelsavepath = mainadd + '/Models/tickclass0.h5'


    best_f1 = -1
    patience = 0
    while patience<5:
        firsttime = time.time()
        hist = f_c.fit(trainx_c, trainy_c, batch_size=batchsize, verbose=0, epochs=1, validation_data=(valx_c,valy_c))
        endtime = time.time()

        valpred = f_c.predict(valx_c,verbose=0)
        # Here is where I calculate f1 score to determine when to save my model...
        current_f1 = np.zeros((winnum))
        for ti in range(winnum):
            for f in range(stocks):
                tempvalf1 = metrics.f1_score(y_true=np.argmax(valy_c[f][:,ti,:], axis=1),
                                               y_pred=np.argmax(valpred[f][:,ti,:], axis=1), average=None)
                current_f1[ti]+= np.average(tempvalf1)/(stocks)


        #current_val = hist.history['val_loss'][0]
        print('')
        print('Window ' + str(0))
        print('Sector ' + str(0))
        print('Epoch Took %.3f Seconds' % (endtime - firsttime))
        print('Train Loss is ' + str(hist.history['loss'][0]))
        print('Validation Loss is ' + str(hist.history['val_loss'][0]))
        print('F1 Score is %.3f' %np.mean(current_f1))
        if np.mean(current_f1)>best_f1:
            best_f1 = np.mean(current_f1)
            mu = best_f1
            patience = 0
            f_c.save_weights(modelsavepath)
            print('New Saved Model')

            # Save model
        else:
            patience+=1


    num_tests = 30
    # Load the best classifier model
    f_c.load_weights(modelsavepath)

    pivot_len=12
    pivot_point=0
    for k in range(num_tests):

        pivot_point += 12 # (12 = hour)

        pivot_data = data[end:end+pivot_point]
        pivot_labels = [labels[end:end+pivot_point,:,i,:] for i in range(5)]
        # First, we need to classify and predict on new data. I'm not sure whether I should do this in chunks or not.
        # Probably, let's just do every two hour chunks for now, this will likely be changed in the final model..

        hist_ae = f_ae.evaluate(pivot_data,pivot_data,verbose=0)
        pred_c = f_c.precict(pivot_data,verbose=0)


        # Here is where I calculate f1 score to determine when to save my model...
        current_f1 = np.zeros((winnum))
        for ti in range(winnum):
            for f in range(stocks):
                tempvalf1 = metrics.f1_score(y_true=np.argmax(pivot_labels[f][:,ti,:], axis=1),
                                               y_pred=np.argmax(pred_c[f][:,ti,:], axis=1), average=None)
                current_f1[ti]+= np.average(tempvalf1)/(stocks)

        rho_hat_future = alpha_rho_future*rho + (1-alpha_rho_future)*hist_ae.history['val_loss'][0]
        mu_hat = alpha_mu*mu + (1-alpha_mu)*np.mean(current_f1)

        if np.abs(rho_hat_future - rho)>sigma_rho or mu - mu_hat>sigma_mu:
            T_e = end + pivot_point-12
            T_i = T_e - T_w
            retrain_ae_data = data[T_i:T_e]

            # retrain the autoencoder model, then use results from this to retrain the classifier model
            pass



    return

if __name__ == '__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])

    main(weekcount,ticknum,winnum)