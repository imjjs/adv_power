"""
Created on Fri Jan 11 15:41:21 2019

@author: xingyu
get a pretrained model
"""

from __future__ import print_function

import urllib.request

from keras.models import load_model

class GetPretrainedModel():
    
    def execute(self):       

        modelname = 'lstmReal150_3_minmax.h5'
        
        path = '/home/dustin/Downloads/power_predict/new109/powerData/'
        
        modelname=path+modelname       

        
        model = load_model(modelname)
        

        return model