# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def genXY(loadHourMeters,loadHourTotal):
    seq_len = 24
    sampleNum= len(loadHourTotal)-seq_len
    X_data = np.zeros([sampleNum,seq_len,loadHourMeters.shape[1]])
    Y_data = np.zeros(sampleNum)
    for i in range(sampleNum):
        X_data[i,:,:] = loadHourMeters[i:i+seq_len,:].reshape([1,seq_len,loadHourMeters.shape[1]])
        Y_data[i] = loadHourTotal[i+seq_len]
    return X_data,Y_data

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
#Group = gen3Group(X_meter,X_temp,Y_data,split_num)

def genGroup(X_meter,X_temp,Y_data,split_num):
    num_meter = X_meter.shape[2]
    num_dif = int(num_meter/split_num)
    Group = []
    #X_0 = X_meter[:,:,0:num_dif]
    #Group.append(X_0)
   
    for i in range(split_num):
        ind = np.ones((num_meter,), bool)
        ind[num_dif*i:num_dif*(i+1)] = False
        Group_i = X_meter[:,:,ind]
        #Group_i = X_meter[:,:,num_dif*i:num_dif*(i+1)]
        Group_i = np.concatenate((Group_i,X_temp),axis=2)
        Group.append(Group_i)
    return Group
    
    
def getDataGroups(split_num):
    tempHour= np.load('tempHour.npy')
    #loadHourTotal=  np.load('loadHourTotal.npy')
    loadHourTotal=  np.load('loadHourTotalReal.npy')
    loadHourMeters = np.load('loadHourMeters.npy')  
    
    shifted_value = loadHourTotal.mean()
    loadHourTotal -= shifted_value
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    #datMat = np.concatenate((loadHourMeters,tempHour.reshape(len(tempHour),1)),axis=1)
    scaledMeter = scaler.fit_transform(loadHourMeters)
    scaledTemp = scaler.fit_transform(tempHour.reshape(len(tempHour),1))

    X_meter,Y_data = genXY(scaledMeter,loadHourTotal)
    X_temp,Y_data = genXY(scaledTemp,loadHourTotal)
    
    Group = genGroup(X_meter,X_temp,Y_data,split_num)
    
    return Group, Y_data

def trainGroup(X_data,Y_data):
    train_row = int(round(0.9 * Y_data.shape[0]))
    
    train_X = X_data[:train_row,:,:]
    train_y = Y_data[:train_row]
    test_X = X_data[train_row:,:,:]
    test_y = Y_data[train_row:]
    return train_X,train_y,test_X,test_y
    