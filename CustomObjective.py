import tensorflow as tf
import keras.backend as K

def mdharmo_crossentropy(y_true, y_pred):

    lam = K.constant(0.5,dtype='float32',shape=0)
    #sig_constant = K.consant()
    #my_sig = tf.nn.sigmoid()

    #obj = K.categorical_crossentropy(y_true,y_pred)+lam*K.categorical_crossentropy(y_true,y_pred)

    return K.categorical_crossentropy(lam*y_true, y_pred)
