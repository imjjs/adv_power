# Editing "FSGMgen" Implementation
# 
# The 'execute' method will be called when the operation is run
from __future__ import print_function
import keras


import numpy as np
#import matplotlib.pyplot as plt
from keras import metrics
#from keras.utils.np_utils import to_categorical
from keras import backend as K

# Create a tensorflow session
#sess = K.get_session()
#print(sess.list_devices)

class FSGMgen():
    def __init__(self, eps=0.3):
        self.eps = eps
        return

    def execute(self, model,data):
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
 
        
        (x_test, y_test) = data

        
        ## Fast Gradient Sign Method Trial
        #eps=0.2
        print("Generating adversarial examples using FGSM method: ")
        adv_x = multiple_FGSM(model, x_test,y_test,self.eps)
        

        
        dat_adv= (adv_x,y_test)
        
        #accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
        #print('Normal Data Test accuracy:', accuracy)

        
        return dat_adv

        
        

