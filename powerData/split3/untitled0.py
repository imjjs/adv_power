#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:21:44 2019

@author: dustin
"""

import pickle

with open("listArr.txt", "rb") as fp:   # Unpickling
    listArr = pickle.load(fp)
#with open("listMax.txt", "rb") as fp:   #Pickling
#    listMax = pickle.load(fp)
with open("listMin.txt", "rb") as fp:   #Pickling
    listMin = pickle.load(fp)
with open("listMean.txt", "rb") as fp:   #Pickling
    listMean = pickle.load(fp)
with open("listSort.txt", "rb") as fp:   #Pickling
    listSort = pickle.load(fp)



def deviation(original, adv):
    assert(len(original) == len(adv))
    ret = []
    for idx in range(len(original)):
        rd = (adv[idx] - original[idx]) / original[idx]
        #print('original: ' + str(original[idx]) + ', adv: ' + str(adv[idx]) + ', deviation: ' + str(rd))
        ret.append(rd)
    return ret

from sklearn.metrics import mean_squared_error
def getMSE(test_y,predict_y):
    test_mse = mean_squared_error(test_y,predict_y)
    return test_mse

mseSort=[]
deviateSort= []
for resSort in listMin:
    mseSort.append(getMSE(test_y,resSort))
    deviateSort.append(deviation(preMin, resSort))

import matplotlib.pyplot as plt
plt.plot(mseSort )
plt.plot(deviateSort[0])
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
 