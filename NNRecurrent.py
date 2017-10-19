# Author: Mark Harmon
# Purpose: Neural Network on the CME Dataset
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
# Load in my data here.  Don't need to put maximum features like they do...
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# Train then test sequence, interesting....
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
# This is to truncate to make sure the sequence size is always consistent.  This is
# something that I'm going to have to play with a lot in my model...
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
# It looks like this embedding layer is mostly for sets like imbd, where we have words...
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
# This is where I probably need to do a for loop.  I believe that you want to take the prediction and use that
# for the next part of training, though I'm not completely sure on this...
model.fit(x_train, y_train, batch_size=batch_size, epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)