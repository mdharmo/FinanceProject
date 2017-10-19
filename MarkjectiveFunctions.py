from __future__ import absolute_import
import six
from keras import backend as K
from keras.utils.generic_utils import deserialize_keras_object

def categorical_crossentropy(y_true, y_pred,g):
    # I need to have a hyper-parameter mu and a trained parameter g involved at this point...
    # This means that in my layer, I need to make sure that k is learned in some way, and that it is recognized at
    # the absolute final state...

    # Ne

    return K.categorical_crossentropy(y_pred, y_true)+mu*g

