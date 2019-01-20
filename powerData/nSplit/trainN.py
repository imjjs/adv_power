# -*- coding: utf-8 -*-

from getDataGroups import getDataGroups,trainGroup

from keras.models import Sequential
from keras.layers import LSTM,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Lambda
from keras import backend as K

import pickle

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
mseN = []
for split_num in range(3,10):
    for i in range(split_num):
        Group,Y_data = getDataGroups(split_num)
#        grpSvName= 'Group'+str(split_num)+'.txt'
#        with open(grpSvName, "wb") as fp:   #Pickling
#            pickle.dump(Group, fp)
        X_raw = Group[i]
        
        train_X,train_y,test_X,test_y =trainGroup(X_raw,Y_data)
        
        model = createLSTM(rate)
        model.fit(train_X, train_y, epochs=50, batch_size=72)
        savename = 'split'+str(split_num)+'rate'+str(int(rate*20))+'G'+str(i)+'.h5'
        model.save('md/'+savename)
        
        test_mse = model.evaluate(test_X, test_y, verbose=1)
        print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
        mseN.append(test_mse)