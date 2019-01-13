# Editing "FreqRand" Implementation
# 
# The 'execute' method will be called when the operation is run
# Editing "MeterRand" Implementation
# 
# The 'execute' method will be called when the operation is run
# Editing "ScoreModel" Implementation
# 
# The 'execute' method will be called when the operation is run
# Editing "ScoreModel" Implementation
# 
# The 'execute' method will be called when the operation is run

import numpy as np
import keras

from scipy import interpolate
import scipy.ndimage
from scipy.signal import medfilt

class FreqRand():

    def execute(self, data):

        (test_X,test_y) = data
        lenTst= len(test_y)
        print('Number of data points:',lenTst)
        
        X_cp = test_X.copy()
        
        X_cp= medfilt(X_cp,kernel_size=(1,3,1))
        
        
        '''
        seq_length = test_X.shape[1]
        feature_num = test_X.shape[2]
        
        meter_num = feature_num-1
        
        X_rnd = np.random.normal(loc=mu, scale=sigma,size=test_X.shape)
        
        ratio= 0.3
        for i in range(lenTst):
            dropCol = np.random.choice(meter_num, int(meter_num*ratio))
            X_cp[i,:,dropCol]=X_rnd[i,:,dropCol]
        
        
        ratio= 0.5
        dropCol = np.random.choice(meter_num, int(meter_num*ratio))
        
        X_cp[:,:,dropCol]=X_rnd[:,:,dropCol]
        '''
        datRnd =   (X_cp,test_y)  
        
        return datRnd    
