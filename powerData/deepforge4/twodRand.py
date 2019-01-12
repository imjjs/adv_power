# Editing "TwoDRand" Implementation
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

class TwoDRand():

    def execute(self, data):

        (test_X,test_y) = data
        lenTst= len(test_y)
        print('Number of data points:',lenTst)
        
        X_cp = test_X.copy()
        
        mu = test_X.mean()
        sigma = test_X.std()
        
        

        seq_length = test_X.shape[1]
        feature_num = test_X.shape[2]
        
        meter_num = feature_num-1
        
        X_rnd = np.random.normal(loc=mu, scale=sigma,size=test_X.shape)
        
        ratio= 0.3
        
        tmp = np.random.randint(100, size=test_X.shape)
        indx = np.where(tmp<100*ratio)
        
        X_cp[indx[0],indx[1],indx[2]] = X_rnd[indx[0],indx[1],indx[2]] 

        
        '''
        ratio= 0.5
        dropCol = np.random.choice(meter_num, int(meter_num*ratio))
        
        X_cp[:,:,dropCol]=X_rnd[:,:,dropCol]
        '''
        datRnd =   (X_cp,test_y)  
        
        return datRnd    
