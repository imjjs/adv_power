from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from keras.models import Sequential, model_from_yaml
from keras.models import load_model
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import loader
from keras import backend as K
from keras import metrics

from keras.models import Model
from keras.layers.core import Lambda



class Experiment(object):
    def __init__(self, _instances: loader.Instance, _model):
        self.feature_dim = 24 * 110
        self.instances = _instances
        self.model = _model
        self.grad = K.gradients(self.model.output, self.model.input)

    def set_testing_dropout(self, rate):
        # Store the fully connected layers
        layerbefore = self.model.layers[-2]
        predictions = self.model.layers[-1]

        # Create the dropout layers
        testing_dropout = Lambda(lambda x: K.dropout(x, level=rate))

        # Reconnect the layers
        x = testing_dropout(layerbefore.output)
        predictors = predictions(x)

        # Create a new model
        self.model = Model(input=self.model.input, output=predictors)


    def predict(self, inp_X):
        predicted_values = self.model.predict(inp_X)
        num_test_samples = len(predicted_values)
        predicted_values = np.reshape(predicted_values, (num_test_samples, 1))
        return predicted_values

    def predict_test(self):
        # get the predicted values
        inp = self.instances
        test_mse = self.model.evaluate(inp.X_test, inp.y_test, verbose=1)
        print('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (
        test_mse[0], len(inp.y_test)))
        #print(inp.X_test)
        predicted_values = self.predict(inp.X_test)
        return predicted_values

    def attack(self, reverse=1):
        step_length = 0.005
        steps = 100
        ret = self.multiple_IterativeGSM(self.instances.X_test, step_length, steps, reverse)
#        ret = []
        
#        for idx in range(len(self.instances.y_test)):
#            X = self.instances.X_test[idx]
#            adv = self.multiple_IterativeGSM([X], step_length, steps, reverse)[0]
#            ret.append(adv)
        return ret

    def multiple_IterativeGSM(self, X_tst, step_length, steps, reverse):
        model = self.model
        x_copy = X_tst.copy()
        get_grad_values = K.function([model.input], self.grad)
        for step in range(steps):
            # stepLen = step_length * (x_copy.max_value - x_copy.min_value) / steps
            grad_values = get_grad_values([x_copy])[0]
            grad_signs = np.sign(grad_values)  # get sign
            noise = grad_signs * step_length  # noise
            x_copy = x_copy + noise * reverse
        return x_copy

# # plot the results
# fig = plt.figure()
# plt.plot(y_test + shifted_value)
# plt.plot(predicted_values + shifted_value)
# plt.xlabel('Hour')
# plt.ylabel('Electricity load (*1e5)')
# plt.show()
# fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')

# save the result into txt file
# test_result = np.vstack((predicted_values, y_test)) + shifted_value
# np.savetxt('output_load_forecasting_result.txt', test_result)

# %%

## Fast Gradient Sign Method Trial
# eps=0.6
# print("Generating adversarial examples using FGSM method: ")
# X_adv = multiple_FGSM(model, X_test,y_test,eps)




# Iterative Gradient Sign Method Trial


## Adversarial attack evaluation
# noise = X_adv - X_test
# X_original = X_test + shifted_value
# noise_rate = np.sum(noise ** 2, axis=1) ** (0.5) / (np.sum(X_original ** 2, axis=1) ** (0.5))
# print('\nNoise ratio: ', noise_rate.mean())
#
# # evaluate the result
# test_mse = model.evaluate(X_test, y_test, verbose=1)
# print('\nMSE on the original test set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))
#
# # evaluate the result
# test_mse = model.evaluate(X_adv, y_test, verbose=1)
# print('\nMSE on the adv data set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))
#
# # get the predicted values
# predicted_values = model.predict(X_test)
# num_test_samples = len(predicted_values)
# predicted_values = np.reshape(predicted_values, (num_test_samples, 1))
#
# # plot the results
# fig = plt.figure()
# plt.plot(y_test + shifted_value)
# plt.plot(predicted_values + shifted_value)
# plt.xlabel('Hour')
# plt.ylabel('Electricity load (*1e5)')
# plt.show()
# fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')