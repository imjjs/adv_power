# Editing "Train" Implementation
# 
# The 'execute' method will be called when the operation is run
from __future__ import print_function
import keras

from keras import backend as K
#import tensorflow as tf

class preTrain():
    def __init__(self, model, epochs=50, batch_size=72):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        return


    def execute(self, data):
        (x_train, y_train) = data
        model = self.model

        
        model.compile(loss='mse',  #  mse = MSE = mean_squared_error
              optimizer='adam')

        # train the model
        model.fit(x_train, y_train, 
                  batch_size=self.batch_size, 
                  epochs=self.epochs,
                  verbose=1)
        
        
        return model
        