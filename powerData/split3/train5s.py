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
#datMat = np.concatenate((loadHourMeters,tempHour.reshape(len(tempHour),1)),axis=1)
scaledMeter = scaler.fit_transform(loadHourMeters)
scaledTemp = scaler.fit_transform(tempHour.reshape(len(tempHour),1))
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
X_meter,Y_data = genXY(scaledMeter,loadHourTotal)
X_temp,Y_data = genXY(scaledTemp,loadHourTotal)
#%%
#X_data= X_data.reshape(X_data.shape[0],X_data.shape[1],X_data.shape[2],1)


split_num = 3
def genGroup(X_meter,X_temp,Y_data,split_num):
    num_meter = X_meter.shape[2]
    num_dif = int(num_meter/split_num)
    Group = []
    X_0 = X_meter[:,:,0:num_dif]
    
    Group.append(X_0)
    X_1 = X_meter[:,:,(num_meter-num_dif):]
    Group.append(X_1)
    #for i in range(split_num-2):
        
def gen3Group(X_meter,X_temp,Y_data,split_num):
    num_meter = X_meter.shape[2]
    num_dif = int(num_meter/split_num)
    Group = []
    X_0 = X_meter[:,:,0:num_dif*2]
    X_0 = np.concatenate((X_0,X_temp),axis=2)
    Group.append(X_0)
    X_2 = X_meter[:,:,(num_meter-2*num_dif):]
    X_2 = np.concatenate((X_2,X_temp),axis=2)
    Group.append(X_2)
    colIdx =  list(range(num_dif))+list(range(num_meter-num_dif,num_meter))
    X_1 = X_meter[:,:,colIdx]
    X_1 = np.concatenate((X_1,X_temp),axis=2)
    Group.append(X_1)
    return Group
Group = gen3Group(X_meter,X_temp,Y_data,split_num)

#%%
def trainGroup(X_data,Y_data):
    train_row = int(round(0.9 * Y_data.shape[0]))
    
    train_X = X_data[:train_row,:,:]
    train_y = Y_data[:train_row]
    test_X = X_data[train_row:,:,:]
    test_y = Y_data[train_row:]
    return train_X,train_y,test_X,test_y
#%%
from keras.models import Sequential
from keras.layers import LSTM,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Lambda
from keras import backend as K
def createLSTM(rate):
    # design network
    model = Sequential()

    
    model.add(Lambda(lambda x: K.dropout(x, level=rate,
                noise_shape=(24, 1)   )))
    
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

rate = 0.2
for i in range(split_num):
    X_raw = Group[i]
    
    train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
    
    model = createLSTM(rate)
    model.fit(train_X, train_y, epochs=50, batch_size=72)
    savename = 'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
    model.save(savename)
    
    test_mse = model.evaluate(test_X, test_y, verbose=1)
    print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))

#%%
from keras.models import load_model

predicted = []
for i in range(split_num):
    X_raw = Group[i]
    
    train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
    
    savename = 'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
    model = load_model(savename)
    
    
    
    predicted.append( model.predict(test_X) )
#%%
predictedArr = np.concatenate((predicted[0],predicted[1],predicted[2] ),axis=1)

preMax = predictedArr.max(axis=1).reshape(216)
test_mse = mean_squared_error(test_y,preMax)
print ('\nThe MSE max on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    
preMean = predictedArr.mean(axis=1).reshape(216)    
test_mse = mean_squared_error(test_y,preMean)
print ('\nThe MSE mean on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))


predictedArr.sort(axis=1)
preSort = np.zeros(216)
dif = (predictedArr[:,1]-predictedArr[:,0])-(predictedArr[:,2]-predictedArr[:,1])
idx = np.where(dif>0)
preSort[idx] = (predictedArr[idx,2]+predictedArr[idx,1])/2
idx = np.where(dif<=0)
preSort[idx] = (predictedArr[idx,1]+predictedArr[idx,0])/2
test_mse = mean_squared_error(test_y,preSort)
print ('\nThe MSE mean on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))


#rate = 0.2
#model = createLSTM(X_data,rate)
#
#
##%%
## fit network
##history = model.fit(train_X, train_y, epochs=150, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#history = model.fit(train_X, train_y, epochs=50, batch_size=72)
#
## plot history
#import matplotlib.pyplot as plt
#plt.plot(history.history['loss'], label='train')
##plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()
#
##
## evaluate the result
#test_mse = model.evaluate(test_X, test_y, verbose=1)
#print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
#
## get the predicted values
#predicted_values = model.predict(test_X)
#num_test_samples = len(predicted_values)
#predicted_values = np.reshape(predicted_values, (num_test_samples,1))
##import matplotlib.pyplot as plt
## plot the results
#def plotPred(predicted_values,y_test):
#    fig = plt.figure()
#    plt.plot(y_test )
#    plt.plot(predicted_values)
#    plt.xlabel('Hour')
#    plt.ylabel('Electricity load (*1e5)')
#    plt.show()
#    #fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')
#plotPred(predicted_values,test_y)  


#%%
'''
his_record = []
mse_record = []
for i in range(0,100,5):
    rate = i/100
    model = createLSTM(X_data,rate)
    history = model.fit(train_X, train_y, epochs=50, batch_size=72)
    his_record.append(history)
    test_mse = model.evaluate(test_X, test_y, verbose=1)
    print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    mse_record.append(test_mse)
    savename = 'rate'+str(i)+'.h5'
    model.save(savename)
''' 
#%%
#mse_record = plt.plot(y_test )