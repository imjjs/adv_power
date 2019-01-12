"""
Created on Fri Jan 11 09:17:15 2019

@author: dustin
"""

import numpy as np

tempHour= np.load('tempHour.npy')
#loadHourTotal=  np.load('loadHourTotal.npy')
loadHourTotal=  np.load('loadHourTotalReal.npy')
loadHourMeters= np.load('loadHourMeters.npy')  




shifted_value = loadHourTotal.mean()
loadHourTotal -= shifted_value

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# MinMax normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Add the temperature feature. So now the input dimension becomes (24,110)
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
#seq_len = 24
#sampleNum= len(loadHourTotal)-seq_len
#X_data = np.zeros([sampleNum,seq_len*loadHourMeters.shape[1]])
#Y_data = np.zeros(sampleNum)
#for i in range(sampleNum):
#    X_data[i,:] = loadHourMeters[i:i+seq_len,:].reshape([1,seq_len*loadHourMeters.shape[1]])
#    Y_data[i] = loadHourTotal[i+seq_len]

#%%
#X_data= X_data.reshape(X_data.shape[0],X_data.shape[1],X_data.shape[2],1)
train_row = int(round(0.9 * Y_data.shape[0]))

train_X = X_data[:train_row,:,:]
train_y = Y_data[:train_row]
test_X = X_data[train_row:,:,:]
test_y = Y_data[train_row:]

#%%
from keras.models import load_model

modelname = 'lstmReal150_3_minmax.h5'
model = load_model(modelname)
model.summary()

#%%
import matplotlib.pyplot as plt
# evaluate the result
test_mse = model.evaluate(test_X, test_y, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))

# get the predicted values
predicted_values = model.predict(test_X)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
#import matplotlib.pyplot as plt
# plot the results
def plotPred(predicted_values,y_test):
    fig = plt.figure()
    plt.plot(y_test )
    plt.plot(predicted_values)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load (*1e5)')
    plt.show()
    #fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')
plotPred(predicted_values,test_y)  