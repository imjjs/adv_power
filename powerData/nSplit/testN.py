# -*- coding: utf-8 -*-

from getDataGroups import getDataGroups,trainGroup
from keras import backend as K
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error


def getPreArr(listPre,split_num):
    predictedArr = listPre[0]
    for i in range(1,split_num):
        predictedArr = np.concatenate((predictedArr,predicted[i]),axis=1)
    return predictedArr

rate = 0.2
mseN = []
listArr = []

for split_num in range(3,10):
    predicted = []
    for i in range(split_num):
        Group,Y_data = getDataGroups(split_num)
#        grpSvName= 'Group'+str(split_num)+'.txt'
#        with open(grpSvName, "wb") as fp:   #Pickling
#            pickle.dump(Group, fp)
        X_raw = Group[i]
        
        train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
        
        savename = 'split'+str(split_num)+'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
        model = load_model('md/'+savename)
        
        predicted.append( model.predict(test_X) )
        
    predictedArr = getPreArr(predicted,split_num)
    length = len(predictedArr)    
            

        
        
        
    listArr.append(predictedArr)

        #listSort.append(preSort)
#%%
def getPreSort(inArr,test_y):
    arr = inArr.copy()
    arr.sort(axis=1)
    preMean = arr[:,1:-1].mean(axis=1)
    return preMean

def getPreMax(predictedArr,test_y):
    length = len(predictedArr)
    preMax = predictedArr.max(axis=1).reshape(length)
    return preMax

def getPreMin(predictedArr,test_y):
    length = len(predictedArr)
    preMin = predictedArr.min(axis=1).reshape(length)
    return preMin

def getPreMean(predictedArr,test_y):
    length = len(predictedArr)
    preMean = predictedArr.mean(axis=1).reshape(length)
    return preMean

listMax = []
listMin = []
listMean = []
listSort = []

mseMax = []
mseMin = []
mseMean =[]
mseSort =[] 
for split_num in range(3,10):
    predictedArr = listArr[split_num-3]
    
    preMax = getPreMax(predictedArr,test_y)
    test_mse = mean_squared_error(test_y,preMax)
    mseMax.append(test_mse)
    print ('\nThe MSE max on test data set is %.3f over %d test samples.' % (test_mse, length))

    preMin = getPreMin(predictedArr,test_y)
    test_mse = mean_squared_error(test_y,preMin)
    mseMin.append(test_mse)
    print ('\nThe MSE min on test data set is %.3f over %d test samples.' % (test_mse, length))
        
    preMean = getPreMean(predictedArr,test_y) 
    test_mse = mean_squared_error(test_y,preMean)
    mseMean.append(test_mse)
    print ('\nThe MSE mean on test data set is %.3f over %d test samples.' % (test_mse, length))
    
    preSort = getPreSort(predictedArr,test_y) 
    test_mse = mean_squared_error(test_y,preSort)
    mseSort.append(test_mse)
    print ('\nThe MSE sort on test data set is %.3f over %d test samples.' % (test_mse, length))
    
    listMax.append(preMax)
    listMin.append(preMin)
    listMean.append(preMean)
    listSort.append(preSort)
    
#%%
import matplotlib.pyplot as plt
splits = np.arange(3, 10)
fig, ax = plt.subplots()
ax.plot(splits,mseMax,label='Max' )
ax.plot(splits,mseMin,label='Min' )
ax.plot(splits,mseMean,label='Mean')
ax.plot(splits,mseSort,label='Sort')
#legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
ax.legend()
plt.xlabel('Split Num')
plt.ylabel('MSE')
plt.show()
     

#%%
import pickle
with open("listArrN.txt", "wb") as fp:   #Pickling
    pickle.dump(listArr, fp)