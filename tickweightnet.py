# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset (Simple Recurrent Model with GRU)
# Walk Forward Method incoming...

from __future__ import print_function
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, GRU, Dense
from keras.models import Model,load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle as pkl
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import metrics
import time

def build_rnn():
    main_input = Input(shape=(5,5,), dtype='float32', name='main_input')
    gru1 = GRU(512,return_sequences=True)(main_input)
    gru2 = GRU(512)(gru1)
    # I'm looping right here, but this may be overly complicated
    out = [Dense(5, activation='softmax', name=('main_output' + str(i)))(gru2) for i in range(5)]
    model = Model(main_input, out)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def make_class_weights(classlab):

    for i in range(5):
        temp = np.max(np.sum(classlab[0],axis=0))/(np.sum(classlab[0],axis=0)+1)
        if i ==0:
            all = temp
        else:
            all = np.concatenate((all,temp))

    # Long and annoying, but easy to read. Find a better way to do this later...
    class_weight = [[{0: all[0],
                    1: all[1],
                    2: all[2],
                    3: all[3],
                    4: all[4]}],[{0: all[5],
                    1: all[6],
                    2: all[7],
                    3: all[8],
                    4: all[9]}],[{0: all[10],
                    1: all[11],
                    2: all[12],
                    3: all[13],
                    4: all[14]}],[{0: all[15],
                    1: all[16],
                    2: all[17],
                    3: all[18],
                    4: all[19]}],[{0: all[20],
                    1: all[21],
                    2: all[22],
                    3: all[23],
                    4: all[24]}]]

    return class_weight

# Train model here
def main():
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train05.pkl'

    data,labels = pkl.load(open(address,'rb'))

    data = (data - np.mean(data,axis=0))/np.std(data,axis=0)
    model = build_rnn()
    epochs = 100
    month_len = 4*8064
    year_len = 8064*12
    week = int((month_len/4.)/4.)
    numtests = 30


    num_its = int(4*len(data)/month_len)
    print(num_its)
    modelsavepath = '/home/mharmon/FinanceProject/ModelResults/tick/tickmodel05.hdf5'

    beg = 0
    end = month_len
    pngcount=0
    f1store = np.zeros((numtests,5,5))

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


            class_weight = make_class_weights(trainy)

            best_loss = 10000
            patience = 0
            while patience<5:
                firsttime=time.time()
                hist = model.fit(trainx, trainy, batch_size=64, verbose=0, epochs=1, validation_data=(valx,valy),
                                 class_weight=class_weight)
                endtime=time.time()
                current_val = hist.history['val_loss'][0]
                print('')
                print('Window ' + str(i))
                print('Round ' + str(j))
                print('Epoch Took "%.3f Seconds' %(endtime-firsttime))
                print('Train Loss is ' + str(hist.history['loss'][0]))
                print('Validation Loss is ' + str(hist.history['val_loss'][0]))
                if current_val<best_loss:
                    best_loss = current_val
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
        eval = model.evaluate(testx,testy,verbose=1)
        # This will need to be done for each softmax function
        # This gathers all of the f1 scores in a reasonable manner...
        for f in range(5):
            f1tempscore = metrics.f1_score(y_true=np.argmax(testy[f], axis=1), y_pred=np.argmax(predictions[f], axis=1),average=None)
            f1where =  np.unique(np.argmax(testy[f],axis=1))
            f1wherestore+=[f1where.tolist()]
            for c in range(len(f1where)):
                f1store[i,f,f1where[c]]=f1tempscore[c]

        f1save = '/home/mharmon/FinanceProject/ModelResults/tick/f1score.npy'
        fwheresave = '/home/mharmon/FinanceProject/ModelResults/tick/fwhere.pkl'
        np.save(f1save, np.array(f1store))
        pkl.dump(f1wherestore,open(fwheresave,'wb'))
        # I should only have 25 histograms for all test runs
        print()
        print('Finished Window ' + str(i))
        print('Training Loss is ' +'%.5f' %model.history['loss'][0])
        print('Validation Loss is ' + '%.5f' %model.history['val_loss'][0])

        figmain = '/home/mharmon/FinanceProject/ModelResults/tick/Figures/HistogramWeek' + str(5+i)
        # I need five histograms per week. It may be useful to also save the actual predictions along
        # with the histogram...

        for h in range(5):

            tempvec1 = np.where(np.argmax(testy[h],axis=1)==0)[0]
            tempvec2 = np.where(np.argmax(testy[h],axis=1)==1)[0]
            tempvec3 = np.where(np.argmax(testy[h],axis=1)==2)[0]
            tempvec4 = np.where(np.argmax(testy[h],axis=1)==3)[0]
            tempvec5 = np.where(np.argmax(testy[h],axis=1)==4)[0]


            if len(tempvec1)>0:
                stock1[h]+=predictions[h][tempvec1,0].tolist()
            if len(tempvec2)>0:
                stock2[h]+= predictions[h][tempvec2,1].tolist()
            if len(tempvec3)>0:
                stock3[h]+=predictions[h][tempvec3,2].tolist()
            if len(tempvec4)>0:
                stock4[h]+=predictions[h][tempvec4,3].tolist()
            if len(tempvec5)>0:
                stock5[h]+=predictions[h][tempvec5,4].tolist()



        # Print histograms at the end I suppose...
        figmain = '/home/mharmon/FinanceProject/ModelResults/tick/Figures/HistogramAllWeeksStock'

        for c in range(5):

            figfinal = figmain +  '1Class'+str(c)+'.png'
            plt.figure(pngcount)
            plt.hist(stock1[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i+1) + ' Weeks For Stock 1' + ' And Class ' + str(c))
            plt.savefig(figfinal)
            plt.close()
            pngcount+=1

            figfinal = figmain + '2Class'+str(c)+'.png'
            plt.figure(pngcount)
            plt.hist(stock2[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i+1) + ' Weeks For Stock 2' + ' And Class ' + str(c))
            plt.savefig(figfinal)
            plt.close()
            pngcount+=1

            figfinal = figmain + '3Class'+str(c)+'.png'
            plt.figure(pngcount)
            plt.hist(stock3[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i+1) + ' Weeks For Stock 3' + ' And Class ' + str(c))
            plt.savefig(figfinal)
            plt.close()
            pngcount+=1

            figfinal = figmain + '4Class'+str(c)+'.png'
            plt.figure(pngcount)
            plt.hist(stock4[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i+1) + ' Weeks For Stock 4' + ' And Class ' + str(c))
            plt.savefig(figfinal)
            plt.close()
            pngcount+=1

            figfinal = figmain + '5Class'+str(c)+'.png'
            plt.figure(pngcount)
            plt.hist(stock5[c])
            plt.xlabel('Bins')
            plt.ylabel('Count')
            plt.title('Histogram of ' + str(i+1) + ' Weeks For Stock 5' + ' And Class ' + str(c))
            plt.savefig(figfinal)
            plt.close()
            pngcount+=1



    return hist

if __name__=='__main__':
    main()