# Editing "GetLocalPowerData" Implementation
# 
# The 'execute' method will be called when the operation is run
# Editing "GetPowerData" Implementation
# 
# The 'execute' method will be called when the operation is run
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:41:21 2019

@author: xingyu 
"""

# Editing "GetMNISTData" Implementation
# 
# The 'execute' method will be called when the operation is run
from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K




import tensorflow as tf

import numpy as np
import urllib.request
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class GetLocalPowerData():
    
    def execute(self):
        
        path = '/home/dustin/Downloads/power_predict/new109/powerData/'
        

        filename1=path+'loadHourMeters.npy'

        loadHourMeters= np.load(filename1) 

        filename2=path+'loadHourTotalReal.npy'

        loadHourTotal= np.load(filename2)


        filename3=path+'tempHour.npy'

        tempHour= np.load(filename3)
        

        shifted_value = loadHourTotal.mean()
        loadHourTotal -= shifted_value
        

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        datMat = np.concatenate((loadHourMeters,tempHour.reshape(len(tempHour),1)),axis=1)
        scaled = scaler.fit_transform(datMat)
        #%%
        def genXY(loadHourMeters,loadHourTotal):
            seq_len = 24
            sampleNum= len(loadHourTotal)-seq_len
            X_data = np.zeros([sampleNum,seq_len,loadHourMeters.shape[1]])
            Y_data = np.zeros(sampleNum)
            for i in range(sampleNum):
                X_data[i,:,:] = loadHourMeters[i:i+seq_len,:].reshape([1,seq_len,loadHourMeters.shape[1]])
                Y_data[i] = loadHourTotal[i+seq_len]
            return X_data,Y_data
        X_data,Y_data = genXY(scaled,loadHourTotal)                
        

        train_row = int(round(0.9 * Y_data.shape[0]))
        
        train_X = X_data[:train_row,:,:]
        train_y = Y_data[:train_row]
        test_X = X_data[train_row:,:,:]
        test_y = Y_data[train_row:]

        training_data = (train_X,train_y)
        testing_data = (test_X,test_y)

        return training_data,testing_data