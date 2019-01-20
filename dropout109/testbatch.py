# -*- coding: utf-8 -*-
import numpy as np

tempHour= np.load('tempHour.npy')
#loadHourTotal=  np.load('loadHourTotal.npy')
loadHourTotal=  np.load('loadHourTotalReal.npy')
loadHourMeters = np.load('loadHourMeters.npy')  

shifted_value = loadHourTotal.mean()
loadHourTotal -= shifted_value

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
datMat = np.concatenate((loadHourMeters,tempHour.reshape(len(tempHour),1)),axis=1)
scaled = scaler.fit_transform(datMat)

#%%
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
#%%
#X_data= X_data.reshape(X_data.shape[0],X_data.shape[1],X_data.shape[2],1)
train_row = int(round(0.9 * Y_data.shape[0]))

train_X = X_data[:train_row,:,:]
train_y = Y_data[:train_row]
test_X = X_data[train_row:,:,:]
test_y = Y_data[train_row:]

#%%
from keras.models import Sequential
from keras.layers import LSTM,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Lambda
from keras import backend as K
def createLSTM(X_data,rate):
    # design network
    model = Sequential()
    
    
    
    model.add(Lambda(lambda x: K.dropout(x, level=rate,
                noise_shape=(X_data.shape[1], 1)   )))

    # layer 1: LSTM
    #model.add(LSTM( input_dim=1, output_dim=50, return_sequences=True))
    #model.add(LSTM(150, input_shape=(X_data.shape[1], X_data.shape[2])
    #       , return_sequences=True))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    # layer 2: LSTM
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    # layer 3: LSTM
    model.add(LSTM(150, return_sequences=False))
    model.add(Dropout(0.2))
    
    #model.add(LSTM(output_dim=30, return_sequences=False))
    model.add(Dense(output_dim=500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=500, activation='relu'))
    model.add(Dropout(0.2))
    # layer 4: dense
    # linear activation: a(x) = x
    #model.add(Dense(output_dim=1, activation='linear'))
    
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

#%%
from keras.models import load_model
his_record = []
mse_record = []
for i in range(0,100,50):
    rate = i/100
    
    
    savename = 'rate'+str(i)+'.h5'
    model = load_model(savename)
    
    test_mse = model.evaluate(test_X, test_y, verbose=1)
    print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    mse_record.append(test_mse)
    