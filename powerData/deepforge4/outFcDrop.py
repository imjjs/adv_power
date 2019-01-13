# Editing "OutFcDropout" Implementation
# 
# The 'execute' method will be called when the operation is run
# Editing "Train" Implementation
# 
# The 'execute' method will be called when the operation is run

from __future__ import print_function
import keras

from keras import backend as K
#import tensorflow as tf
from keras import losses
from keras import optimizers

from keras.models import Sequential
from keras.layers import LSTM,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D

from keras.models import Model
from keras.layers.core import Lambda


class OutFcDropout():
    def __init__(self, rate=0.5):
        self.rate = rate
        return


    def execute(self, model):
        rate = self.rate
        
        loss = model.loss
        optimizer = model.optimizer

        # Store the fully connected layers
        layerbefore = model.layers[-3]
        predictions = model.layers[-1]
        
        # Create the dropout layers
        
        dropout1 = (Lambda(lambda x: K.dropout(x, level=rate)))
        
        # Reconnect the layers
        x = dropout1(layerbefore.output)
        predictors = predictions(x)
        
        # Create a new model
        model = Model(input=model.input, output=predictors)
        
        model.compile(loss=loss,optimizer=optimizer)
                  
        return model