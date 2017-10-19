# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense, RepeatVector, TimeDistributed, Flatten, Permute, merge,Lambda,Reshape,Dropout,add,LSTM
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import pickle as pkl
import numpy as np
import keras.backend as K
from os import listdir
from os.path import isfile, join
from sklearn.metrics import log_loss, f1_score
import time
from imblearn.over_sampling import SMOTE as smote
import sys
from SmoteSampling import smote_conv_new
from keras.layers import Activation
from keras import backend as K
import keras
from keras.layers.convolutional_recurrent import ConvLSTM2D
from SampConv import SamplingConvLSTM2D
from CustomObjective import mdharmo_crossentropy

def build_rnn(seqsize,batchsize,labseq,stocks):
    classes = 5
    # Build Encoder here
    main_input = Input(shape=(4,stocks,seqsize,1), dtype='float32', name='main_input')

    '''
    # This is where we do attention (This may need to change, attention for conv models seems odd).
    a1 = Reshape((4,stocks*seqsize))(main_input)
    a2 = TimeDistributed(Dense(1, activation='tanh'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(stocks*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,stocks,seqsize,1))(a6)
    att = merge([main_input, a7],mode='mul')
    '''
    # Encoder first
    conv1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(main_input)
    bn1 = BatchNormalization()(conv1)
    conv2 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=False)(bn1)
    bn2 = BatchNormalization()(conv2)

    # Decoder here

    rep0 = Flatten()(bn2)
    rep1 = RepeatVector(labseq)(rep0)
    rep2 = Reshape((labseq,stocks,seqsize-2,64))(rep1)
    dec1 = ConvLSTM2D(filters=64,kernel_size=(1,2),return_sequences=True)(rep2)
    bn3 = BatchNormalization()(dec1)
    res = Reshape((labseq,stocks*(seqsize-3)*64))(bn3)

    out = [TimeDistributed(Dense(5, activation='softmax', name=('predict_out' + str(i))))(res) for i in range(stocks)]
    # Now we combine the confidence output and prediction output into a single output...
    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def build_seq2seq(seqsize,batchsize,labseq,stocks,winnum,classes):

    ############################################################################################
    # Here we make the training model...
    latent_dim = 64
    encoder_input = Input(shape=(4,stocks,seqsize,1), dtype='float32', name='encoder_input')

    # This is where we do attention (This may need to change, attention for conv models seems odd).
    a1 = Reshape((4,stocks*seqsize))(encoder_input)
    a2 = TimeDistributed(Dense(1, activation='relu'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(stocks*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,stocks,seqsize,1))(a6)
    att = merge([encoder_input, a7],mode='mul')


    e1 = ConvLSTM2D(latent_dim,kernel_size=(1,2), return_sequences=True)
    e1_out = e1(att)

    e2 = ConvLSTM2D(latent_dim*2,kernel_size=(1,2),return_sequences=False,return_state=True)
    encoder_outputs,state_h,state_c=e2(e1_out)
    # We discard `encoder_outputs` and only keep the states.
    temp1 = Flatten()(state_c)
    temp2 = Flatten()(state_h)
    c_cond = Dense(latent_dim*2,activation='relu')(temp1)
    h_cond = Dense(latent_dim*2,activation='relu')(temp2)

    encoder_states = [h_cond, c_cond]


    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,stocks*classes))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    d1= LSTM(latent_dim*2, return_sequences=True, return_state=True)
    d1_out,_,_ = d1(decoder_inputs,initial_state=encoder_states)

    d2 = LSTM(latent_dim*3,return_sequences=True,return_state=True)
    decoder_outputs,_,_= d2(d1_out)

    decoder_dense = [TimeDistributed(Dense(5, activation='softmax'))]*stocks
    decoder_outputs = [decoder_dense[i](decoder_outputs) for i in range(stocks)]

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_input, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss=mdharmo_crossentropy)

    #################################################################################
    # Define sampling models
    encoder_model = Model(encoder_input, encoder_states)

    decoder_state_input_h_1 = Input(shape=(latent_dim*2,))
    decoder_state_input_c_1 = Input(shape=(latent_dim*2,))
    decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]

    decoder_state_input_h_2 = Input(shape=(latent_dim*3,))
    decoder_state_input_c_2 = Input(shape=(latent_dim*3,))
    decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]

    dec1_out,state_h_1,state_c_1= d1(
        decoder_inputs, initial_state=decoder_states_inputs_1)
    decoder_states_1 = [state_h_1,state_c_1]

    decoder_outputs,state_h_2,state_c_2 = d2(dec1_out,initial_state=decoder_states_inputs_2)
    decoder_states_2 = [state_h_2,state_c_2]
    decoder_outputs = [decoder_dense[i](decoder_outputs) for i in range(stocks)]
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs_1 + decoder_states_inputs_2,
        decoder_outputs+decoder_states_1+decoder_states_2)

    return model,encoder_model,decoder_model

def make_decoder_inputs(datay,stocks,winnum,classes):

    temp_input = np.array(datay)
    temp_input = np.swapaxes(temp_input,0,1)
    temp_input = np.swapaxes(temp_input,1,2)
    temp_input = np.reshape(temp_input,(len(temp_input),winnum,classes*stocks))
    decoder_inputs = np.zeros((len(datay[0]),winnum,stocks*classes))

    # Set the initial 'Go' symbol
    decoder_inputs[:,0,:]=1

    # The rest will be filled in with the correct previous label

    decoder_inputs[:,1:,:] = temp_input[:,:-1,:]

    return decoder_inputs

def decode_sequence(input_seq,encoder_model,decoder_model,stocks,classes,batch_size,winnum):
    # Encode the input as state vectors.
    states_value_1 = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    # (batch_size,sequence length,features)
    target_seq = np.zeros((batch_size, 1, stocks*classes))
    # Populate the first character of target sequence with the start character.
    target_seq[:, 0, :] = 1.

    t1 = np.zeros((batch_size,192)).astype('float32')
    t2 = np.zeros((batch_size,192)).astype('float32')

    states_value_2 = [t1,t2]
    # Sampling loop for a batch of sequences
    test_predictions = [np.zeros((batch_size,winnum,classes))]*stocks
    for i in range(winnum):
        outputs = decoder_model.predict(
            [target_seq] + states_value_1+states_value_2)

        h_2=outputs[-2]
        c_2 = outputs[-1]

        h_1 = outputs[-4]
        c_1 = outputs[-3]

        output=outputs[0:22]


        # Sample a token
        target_seq = np.zeros((batch_size, 1, stocks * classes))
        for j in range(stocks):
            test_predictions[j][:,i,:] = output[j][:,0,:]
            temp_pred = np.argmax(output[j][:,0,:],axis=-1)
            target_seq[:,0,temp_pred+j*5]=1

        # Update states
        states_value_1 = [h_1, c_1]
        states_value_2 = [h_2,c_2]

    return test_predictions


# Train model here
def main(weekcount,ticknum,winnum,large,restart):
    labseq = winnum
    stocks = 22
    classes = 5
    numtests=1 # silly placeholder
    old_numtests=1
    mainadd = '/home/mharmon/FinanceProject/ModelResults/ticknewday'+str(ticknum) + 'win' + str(winnum)
    modeladd = mainadd + '/Models/tickmodel'
    batchsize=64
    seqsize = ticknum
    model,encoder_model,decoder_model = build_seq2seq(seqsize,batchsize,labseq,stocks,winnum,classes)
    month_len = int(8064)
    week = int((month_len)/4.)
    # Need to additionally fit the smotes for training purposes...
    for overall in range(large,4):
        address  = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday'+str(ticknum) + 'win'+str(winnum)+'sector'+str(overall)+'.pkl'

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
            print(old_dates.shape)
            print('New data is shape')
            print(newdates.shape)
            print()
            data = np.vstack((old_data,newdata))
            labels = np.vstack((old_labels,newlabels))
            dates = np.concatenate((old_dates,newdates))

            print(dates.shape)
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
            old_add = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday'+str(ticknum) + 'win'+str(winnum)+'sector'+str(overall-1)+'.pkl'
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
            address  = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday'+str(ticknum) +'win'+str(winnum)+ 'sector'+str(overall-1)+'.pkl'
            print(address)
            old_data,old_labels,old_dates = pkl.load(open(address,'rb'))
            old_data = np.swapaxes(old_data,2,4)
            old_data = np.swapaxes(old_data,2,3)
            address  = '/home/mharmon/FinanceProject/Data/tickdata/trainnewday'+str(ticknum) +'win'+str(winnum)+ 'sector'+str(overall)+'.pkl'
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
        f1store = np.zeros((numtests, stocks, winnum, 5))
        f1wherestore=[]


        # These two cases actually take care of everything
        if weekcount>0:
            print('Loading Model ')
            modelsavepath = mainadd + '/Models/tickmodel'+str(weekcount-1) + 'sector' + str(overall) + '.h5'
            encodersavepath = mainadd + '/Models/encoder'+str(weekcount-1)+'sector'+str(overall)+'.h5'
            decodersavepath = mainadd + '/Models/decoder'+str(weekcount-1)+'sector'+str(overall)+'.h5'

            model.load_weights(modelsavepath)
            encoder_model.load_weights(encodersavepath)
            decoder_model.load_weights(decodersavepath)

            # Load the old f1 scores...
            f1save = mainadd+'/f1score' + 'sector'+str(overall)+'.npy'
            f1avgsave = mainadd + '/f1avg'+'sector'+str(overall)+'.npy'
            fwheresave = mainadd + '/fwhere'+'sector'+str(overall)+'.pkl'
            f1store = np.load(f1save)
            f1testavg = np.load(f1avgsave)
            f1wherestore = pkl.load(open(fwheresave, 'rb'))

            if len(f1store)<numtests:
                for z in range(numtests-len(f1store)):
                    f1store=np.append(f1store,np.zeros((1,stocks,winnum,5)))
                    f1testavg=np.append(f1testavg,np.zeros((1,winnum)))

            f1testavg = np.reshape(f1testavg,(numtests,1))
            f1store = np.reshape(f1store,(numtests,5,winnum,5))

        if weekcount==0 and overall >0:
            print('Loading Model')

            modelsavepath = mainadd + '/Models/tickmodel' + str(old_numtests - 1) + 'sector' + str(overall-1) + '.h5'
            encodersavepath = mainadd + '/Models/encoder' + str(old_numtests - 1) + 'sector' + str(overall-1) + '.h5'
            decodersavepath = mainadd + '/Models/decoder' + str(old_numtests - 1) + 'sector' + str(overall-1) + '.h5'

            model.load_weights(modelsavepath)
            encoder_model.load_weights(encodersavepath)
            decoder_model.load_weights(decodersavepath)

        for i in range(weekcount,numtests):

            modelsavepath = mainadd + '/Models/tickmodel' + str(i) + 'sector' + str(overall) + '.h5'
            encodersavepath = mainadd + '/Models/encoder' + str(i) + 'sector' + str(overall) + '.h5'
            decodersavepath = mainadd + '/Models/decoder' + str(i) + 'sector' + str(overall) + '.h5'

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

                trainx, trainy = smote_conv_new(trainx, trainy,seqsize,winnum,stocks,classes)

                traind = make_decoder_inputs(trainy,stocks,winnum,classes)
                vald = make_decoder_inputs(valy,stocks,winnum,classes)


                best_f1 = -1
                patience = 0
                while patience<5:
                    firsttime = time.time()
                    hist = model.fit([trainx,traind], trainy, batch_size=batchsize, verbose=0, epochs=1, validation_data=([valx,vald],valy))
                    endtime = time.time()

                    #valpred = model.predict(valx,verbose=0)
                    valpred = decode_sequence(valx,encoder_model,decoder_model,stocks,classes,len(valx),winnum)
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
                        model.save_weights(modelsavepath,overwrite=True)
                        encoder_model.save_weights(encodersavepath,overwrite=True)
                        decoder_model.save_weights(decodersavepath,overwrite=True)
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
                encoder_model.load_weights(encodersavepath)
                decoder_model.load_weights(decodersavepath)



            beg += week
            end += week

            for r in range(pred_runs):

                predictions = decode_sequence(testx, encoder_model, decoder_model, stocks, classes, len(testx), winnum)
                #predictions = model.predict(testx, batch_size=32, verbose=1)
                tempred = np.zeros((week,stocks,winnum,5),'float32')
                for p in range(stocks):
                    tempred[:,p,:,:] =predictions[p]


                adds=mainadd + '/Predictions/Sector'+str(overall)+'predictions'+str(i)+'run'+str(r)+'.pkl'
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


