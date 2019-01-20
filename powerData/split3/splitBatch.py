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
from keras import metrics 
from keras.models import load_model
rate = 0.2
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

preMin = predictedArr.min(axis=1).reshape(216)
test_mse = mean_squared_error(test_y,preMin)
print ('\nThe MSE min on the adv data set is %.3f over %d test samples.' % (test_mse, len(test_y)))

    
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
print ('\nThe MSE sort on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))

#%%

def deviation(original, adv):
    assert(len(original) == len(adv))
    ret = []
    for idx in range(len(original)):
        rd = (adv[idx] - original[idx]) / original[idx]
        #print('original: ' + str(original[idx]) + ', adv: ' + str(adv[idx]) + ', deviation: ' + str(rd))
        ret.append(rd)
    return ret

def multiple_IterativeGSM(model, X_tst, step_length, steps, reverse, grad):
    x_copy = X_tst.copy()
    get_grad_values = K.function([model.input], grad)
    for step in range(steps):
        # stepLen = step_length * (x_copy.max_value - x_copy.min_value) / steps
        grad_values = get_grad_values([x_copy])[0]
        grad_signs = np.sign(grad_values)  # get sign
        noise = grad_signs * step_length  # noise
        x_copy = x_copy + noise * reverse
    return x_copy

def multiple_FGSM(model, X_tst,y_tst,eps):
    target = y_tst
    target_variable = K.variable(target)
    loss = metrics.mean_squared_error(model.output, target_variable)
    gradients       = K.gradients(loss, model.input)
    
    
    get_grad_values = K.function([model.input], gradients)
    grad_values     = get_grad_values([X_tst])[0]
        
    grad_signs      = np.sign(grad_values) # get sign
    #epsilon=0.3
    noise           = grad_signs * eps # noise
    adversarial     = X_tst+noise #np.clip(original + noise, 0., 1.) # clipped between(0,1)
    return adversarial

#%% 
def advTest(step_len,Group,X_meter,Y_data):
    GroupAdv = []
    rate = 0.2
    split_num=3
    for i in range(split_num):
        X_raw = Group[i]
        
        train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
        
        savename = 'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
        model = load_model(savename)
        
        # FGSM
        adv_test_x = multiple_FGSM(model, test_X,test_y,step_len)
        
        ## I-GSM
        #grad = K.gradients(model.output, model.input)
        #adv_test_x = multiple_IterativeGSM(model, test_X, .0002, 100, 1, grad)
        
        
        GroupAdv.append( adv_test_x )
    #%% X0 0:72      s1,s2
    #   X2  37:109   s2,s3
    #   X1 0:36+73:109  s1,s3
    num_meter = X_meter.shape[2]
    num_dif = int(num_meter/split_num)
    s1 = list(range(num_dif))
    s2 = list(range(num_dif ,2*num_dif))
    s3 = list(range(num_meter-num_dif,num_meter))
    adv_test_x = np.zeros((216,24,109))
    adv_test_x[:,:,s1] = (GroupAdv[0][:,:,s1] + GroupAdv[2][:,:,s1])/2
    adv_test_x[:,:,s2] = (GroupAdv[0][:,:,s2] + GroupAdv[1][:,:,s1])/2
    adv_test_x[:,:,s3] = (GroupAdv[1][:,:,s2] + GroupAdv[2][:,:,s2])/2
    
    Group = gen3Group(adv_test_x,X_temp[-216:,:,:],Y_data,split_num)
    predicted = []
    for i in range(split_num):
        X_raw = Group[i]
        
        train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
        
        savename = 'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
        model = load_model(savename)
        
        
        
        predicted.append( model.predict(train_X) )
        
    #%%
    
    
    predictedArr = np.concatenate((predicted[0],predicted[1],predicted[2] ),axis=1)
    
    preMax = predictedArr.max(axis=1).reshape(216)
    test_mse = mean_squared_error(test_y,preMax)
    print ('\nThe MSE max on the adv data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    
    preMin = predictedArr.min(axis=1).reshape(216)
    test_mse = mean_squared_error(test_y,preMin)
    print ('\nThe MSE min on the adv data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
     
        
    preMean = predictedArr.mean(axis=1).reshape(216)    
    test_mse = mean_squared_error(test_y,preMean)
    print ('\nThe MSE mean on the adv data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    
    
    predictedArr.sort(axis=1)
    preSort = np.zeros(216)
    dif = (predictedArr[:,1]-predictedArr[:,0])-(predictedArr[:,2]-predictedArr[:,1])
    idx = np.where(dif>0)
    preSort[idx] = (predictedArr[idx,2]+predictedArr[idx,1])/2
    idx = np.where(dif<=0)
    preSort[idx] = (predictedArr[idx,1]+predictedArr[idx,0])/2
    test_mse = mean_squared_error(test_y,preSort)
    print ('\nThe MSE sort on the adv data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
    
    return predictedArr,preMax,preMin,preMean,preSort
#step_len= 0.001
#advTest(step_len,Group,X_meter,Y_data)

listArr = []
listMax = []
listMin = []
listMean = []
listSort = []
for i in range(20):
    print(i)
    step_len = (i+1)*0.005
    predictedArr,preMax,preMin,preMean,preSort = advTest(step_len,Group,X_meter,Y_data)
    listArr.append(predictedArr)
    listMax.append(preMax)
    listMin.append(preMin)
    listMean.append(preMean)
    listSort.append(preSort)

#%%
import pickle
with open("listArrFGSM.txt", "wb") as fp:   #Pickling
    pickle.dump(listArr, fp)
with open("listMaxFGSM.txt", "wb") as fp:   #Pickling
    pickle.dump(listMax, fp)
with open("listMinFGSM.txt", "wb") as fp:   #Pickling
    pickle.dump(listMin, fp)
with open("listMeanFGSM.txt", "wb") as fp:   #Pickling
    pickle.dump(listMean, fp)
with open("listSortFGSM.txt", "wb") as fp:   #Pickling
    pickle.dump(listSort, fp)
