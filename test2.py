from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix


# random seed
np.random.seed(1234)

# load raw data
df_raw = pd.read_csv('data_load.csv', header=None)
# numpy array
df_raw_array = df_raw.values
# daily load
# list_daily_load = [df_raw_array[i,:] for i in range(0, len(df_raw)) if i % 24 == 0]
# hourly load (23 loads for each day)
list_hourly_load = [df_raw_array[i, 5] / 100000 for i in range(0, len(df_raw)) if i % 24 != 0]
# the length of the sequnce for predicting the future value
sequence_length = 25

# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)

# shift all data by mean
matrix_load = np.array(matrix_load)
shifted_value = matrix_load.mean()
matrix_load -= shifted_value
print("Data  shape: ", matrix_load.shape)

# split dataset: 90% for training and 10% for testing
train_row = int(round(0.9 * matrix_load.shape[0]))
train_set = matrix_load[:train_row, :]

# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
# the test set
X_test = matrix_load[train_row:, :-1]
y_test = matrix_load[train_row:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.core import Lambda

'''
https://stackoverflow.com/questions/47787011/how-to-disable-dropout-while-prediction-in-keras
Maybe useful:
https://github.com/keras-team/keras/issues/9412
'''
# build the model
seq_length = sequence_length - 1

model = Sequential()
model.add(Conv1D(seq_length, 3, activation='relu', input_shape=(seq_length, 1)))
model.add(Conv1D(seq_length, 3, activation='relu'))
model.add(MaxPooling1D(3))
# model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
model.add(Conv1D(seq_length * 2, 3, activation='relu'))
model.add(Conv1D(seq_length * 2, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(seq_length * 7, activation='relu'))
# model.add(Dropout(0.3))   #0.189,0.373
# model.add(Lambda(lambda x: K.dropout(x, level=0.3)))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse',  # mse = MSE = mean_squared_error
              optimizer='rmsprop',
              metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, batch_size=512, nb_epoch=50, validation_split=0.05, verbose=1)

# evaluate the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))


# plot the results
def plotPred(predicted_values, y_test, shifted_value):
    fig = plt.figure()
    plt.plot(y_test + shifted_value)
    plt.plot(predicted_values + shifted_value)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load (*1e5)')
    plt.show()
    # fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')


plotPred(predicted_values, y_test, shifted_value)
# save the result into txt file
# test_result = np.vstack((predicted_values, y_test)) + shifted_value
# np.savetxt('output_load_forecasting_result.txt', test_result)

# %%
from keras import backend as K
from keras import metrics

K.set_learning_phase(0)


def multiple_FGSM(model, X_tst, y_tst, eps):
    target = y_tst
    target_variable = K.variable(target)
    loss = metrics.mean_squared_error(model.output, target_variable)
    gradients = K.gradients(loss, model.input)

    get_grad_values = K.function([model.input], gradients)
    grad_values = get_grad_values([X_tst])[0]

    grad_signs = np.sign(grad_values)  # get sign
    # epsilon=0.3
    noise = grad_signs * eps  # noise
    adversarial = X_tst + noise  # np.clip(original + noise, 0., 1.) # clipped between(0,1)
    return adversarial


# Fast Gradient Sign Method Trial
eps = 0.2
print("Generating adversarial examples using FGSM method: ")
X_adv = multiple_FGSM(model, X_test, y_test, eps)


def multiple_IterativeGSM(model, X_tst, y_tst, step_length, steps):
    x_copy = X_tst.copy()
    target = y_tst
    target_variable = K.variable(target)
    loss = metrics.mean_squared_error(model.output, target_variable)
    gradients = K.gradients(loss, model.input)

    get_grad_values = K.function([model.input], gradients)

    for step in range(steps):
        # stepLen = step_length * (x_copy.max_value - x_copy.min_value) / steps
        grad_values = get_grad_values([x_copy])[0]
        grad_signs = np.sign(grad_values)  # get sign
        noise = grad_signs * step_length  # noise
        x_copy = x_copy + noise
    return x_copy


## Iterative Gradient Sign Method Trial
# step_length = 0.0005
# steps = 100
# print("Generating adversarial examples using I-GSM method: ")
# X_adv = multiple_IterativeGSM(model, X_test,y_test, step_length, steps)


## Adversarial attack evaluation
noise = X_adv - X_test
X_original = X_test + shifted_value
noise_rate = np.sum(noise ** 2, axis=1) ** (0.5) / (np.sum(X_original ** 2, axis=1) ** (0.5))
print('\nNoise ratio: ', noise_rate.mean())

# evaluate the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print('\nMSE on the original test set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# evaluate the result
test_mse = model.evaluate(X_adv, y_test, verbose=1)
print('\nMSE on the I-GSM adv data set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# get the predicted values
predicted_values = model.predict(X_adv)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))

# plot the results
plotPred(predicted_values, y_test, shifted_value)
# fig = plt.figure()
# plt.plot(y_test + shifted_value)
# plt.plot(predicted_values + shifted_value)
# plt.xlabel('Hour')
# plt.ylabel('Electricity load (*1e5)')
# plt.show()
# fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')


# %% Randomization Defense
# https://github.com/cihangxie/NIPS2017_adv_challenge_defense
from scipy import interpolate


def multiple_Randomize(X_tst):
    #    upB = np.max(X_tst)
    #    lowB = np.min(X_tst)
    #    noiseB = (upB-lowB)/20
    dif = 1

    seq_length = X_tst.shape[1]
    step_int = seq_length / (seq_length - dif)
    x_new = np.arange(0, seq_length - dif, step_int)
    x_old = np.arange(0, seq_length)

    X_rnd = np.zeros(X_tst.shape)

    for i in range(len(X_tst)):
        tmp = X_tst[i, :, 0]  # + np.random.normal(0,noiseB,seq_length)

        f = interpolate.interp1d(x_old, tmp, kind='quadratic')
        tmp2 = f(x_new)

        X_rnd[i, (seq_length - len(tmp2)):, 0] = tmp2

    return X_rnd


X_rnd = multiple_Randomize(X_test)
X_rndAdv = multiple_FGSM(model, X_rnd, y_test, eps)
# X_rndAdv =  multiple_IterativeGSM(model, X_rnd,y_test, step_length, steps)
# evaluate the result
test_mse = model.evaluate(X_rnd, y_test, verbose=1)
print('\nMSE on the rand Pad set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

test_mse = model.evaluate(X_rndAdv, y_test, verbose=1)
print('\nMSE on the adv Pad set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# %% Randomization Defense from frequency domain
from numpy import fft


def datFFT(fx):
    n = len(fx)  # Number of data points
    dx = 1

    Fk = fft.rfft(fx) / n  # Fourier coefficients (divided by n)
    nu = fft.rfftfreq(n, dx)  # Natural frequencies
    # Fk = fft.fftshift(Fk) # Shift zero freq to center
    # nu = fft.fftshift(nu) # Shift zero freq to center
    f, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(nu, np.real(Fk))  # Plot Cosine terms
    ax[0].set_ylabel(r'$Re[F_k]$', size='x-large')
    ax[1].plot(nu, np.imag(Fk))  # Plot Sine terms
    ax[1].set_ylabel(r'$Im[F_k]$', size='x-large')
    ax[2].plot(nu, np.absolute(Fk) ** 2)  # Plot spectral power
    ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size='x-large')
    ax[2].set_xlabel(r'$\widetilde{\nu}$', size='x-large')
    plt.show()
    return Fk, nu


def rndFFT(fx):
    n = len(fx)  # Number of data points

    Fk = fft.rfft(fx) / n  # Fourier coefficients (divided by n)
    Fk[-1] /= 2
    Freal = np.real(Fk)
    Fimg = np.imag(Fk)
    rndidx = np.random.randint(int(n / 2), size=2)
    Freal[rndidx[0]] = Freal[rndidx[0]] * 0.9
    Fimg[rndidx[1]] = Freal[rndidx[1]] * 0.9
    Fnew = Freal + 1j * Fimg
    fnew = fft.irfft(Fnew, 2 * n)[::2] * 2 * n
    return fnew


def multipleRndFFT(X_tst):
    X_rnd = np.zeros(X_tst.shape)

    for i in range(len(X_tst)):
        tmp = X_tst[i, :, 0]  # + np.random.normal(0,noiseB,seq_length)
        X_rnd[i, :, 0] = rndFFT(tmp)

    return X_rnd


X_rnd = multipleRndFFT(X_test)
# X_rndAdv =  multiple_IterativeGSM(model, X_rnd,y_test, step_length, steps)
X_rndAdv = multiple_FGSM(model, X_rnd, y_test, eps)
# evaluate the result
test_mse = model.evaluate(X_rnd, y_test, verbose=1)
print('\nMSE on the rand FFT set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

test_mse = model.evaluate(X_rndAdv, y_test, verbose=1)
print('\nMSE on the adv FFT set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# i=3
# tst0=X_test[i,:,0]
# Fk,nu=datFFT(tst0)
# Fk[-1] /= 2
# dif = tst0- rndFFT(tst0)