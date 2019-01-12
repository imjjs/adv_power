# Editing "InputRand" Implementation
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

class InputRand():

    def execute(self, data):

        (test_X,test_y) = data
        lenTst= len(test_y)
        print('Number of data points:',lenTst)
        
        dif = 1
    
        seq_length = test_X.shape[1]
        feature_num = test_X.shape[2]
        #step_int = seq_length/(seq_length-dif)
        #x_new = np.arange(0, seq_length-dif, step_int)
        #x_old = np.arange(0, seq_length)
        
        mu = test_X.mean()
        sigma = test_X.std()
        
        X_rnd= np.zeros(test_X.shape)
        zoomfact= float(seq_length-dif)/seq_length
        out=scipy.ndimage.interpolation.zoom(input=test_X, zoom=(1,zoomfact,1), order = 2)
        
        X_rnd[:,dif:,:]=out
        X_rnd[:,0:dif,:]= np.random.normal(loc=mu, scale=sigma,
                size=(lenTst,dif,feature_num))
        
        datRnd =   (X_rnd,test_y)  
        
        return datRnd    
