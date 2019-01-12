# Editing "FSGMgen" Implementation
# 
# The 'execute' method will be called when the operation is run
from __future__ import print_function
#import keras


import numpy as np
#import matplotlib.pyplot as plt
#from keras import metrics
#from keras.utils.np_utils import to_categorical
from keras import backend as K

# Create a tensorflow session
#sess = K.get_session()
#print(sess.list_devices)

class IGSMgen():
    def __init__(self, step_length=0.005, steps=100, reverse=1):
        self.step_length = step_length
        self.steps = steps
        self.reverse = reverse
        return

    def execute(self, data, model ):
       
        def multiple_IterativeGSM( model, X_tst, step_length, steps, reverse):
            gradients       = K.gradients(model.output, model.input)       
            get_grad_values = K.function([model.input], gradients)            
            
            x_copy = X_tst.copy()

            for step in range(steps):
                # stepLen = step_length * (x_copy.max_value - x_copy.min_value) / steps
                grad_values = get_grad_values([x_copy])[0]
                grad_signs = np.sign(grad_values)  # get sign
                noise = grad_signs * step_length  # noise
                x_copy = x_copy + noise * reverse
            return x_copy
 
        
        (x_test, y_test) = data

        
        ## Iterative Gradient Sign Method Trial
        #eps=0.2
        print("Generating adversarial examples using Iterative-GSM Method: ")
        adv_x = multiple_IterativeGSM(model, x_test, self.step_length, 
                                      self.steps, self.reverse)
        
      
        dat_adv= (adv_x,y_test)
        
        #accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
        #print('Normal Data Test accuracy:', accuracy)

        
        return dat_adv

        
        


