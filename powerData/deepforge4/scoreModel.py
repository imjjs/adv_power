# Editing "ScoreModel" Implementation
# 
# The 'execute' method will be called when the operation is run

import numpy as np
import keras
class ScoreModel():

    def execute(self, model, data):
        print('Model Under Test:')
        model.summary()
        
        
#        training_data = (train_X,train_y)
#        testing_data = (test_X,test_y)
        (test_X,test_y) = data
        lenTst= len(test_y)
        print('Number of data points:',lenTst)
        
        
        # evaluate the result
        test_mse = model.evaluate(test_X, test_y, verbose=1)
        print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
        
        # get the predicted values
        predicted_values = model.predict(test_X)
        num_test_samples = len(predicted_values)
        predicted_values = np.reshape(predicted_values, (num_test_samples,1))
        

        return test_mse