import loader
import model
import numpy
import pickle
import powerData.deepforge4.getPowerData
from keras import backend as K
from keras.models import load_model, Model
from keras.layers.core import Lambda
from sklearn.metrics import mean_squared_error

def set_testing_dropout(model, rate):
    # Store the fully connected layers
    layerbefore = model.layers[-2]
    predictions = model.layers[-1]

    # Create the dropout layers
    testing_dropout = Lambda(lambda x: K.dropout(x, level=rate))

    # Reconnect the layers
    x = testing_dropout(layerbefore.output)
    predictors = predictions(x)

    # Create a new model
    new_model = Model(input=model.input, output=predictors)
    #new_model.summary()
    return new_model



if __name__ == '__main__':
    dataobj = powerData.deepforge4.getPowerData.GetPowerData()
    data = dataobj.execute()
    print(data[3].shape)
    _, _, test_x, test_y = data
    test_x_list = [test_x[idx] for idx in range(len(test_x))]
    model = load_model('powerData/lstm150fc500.h5')
    # model.summary()
    # res_model = set_testing_dropout(model, .1)
    grad = K.gradients(model.output, model.input)
    from multiprocessing import Pool
    import l_0
    import l_inf
    l0_warp = lambda t: l_0.l0_attack(model, t, -1, grad, [q * 50 for q in range(1, 40, 2)])
    l_inf_warp = lambda t: l_inf.l_inf_attack(model, t, 1e-4, -1, grad, [10*q for q in range(1, 200, 2)])
    with Pool(16) as p:
        l_0_ret = p.map(l0_warp, test_x_list)
        l_inf_ret = p.map(l_inf_warp, test_x_list)

    with open('l0.pk', 'wb') as pk:
        pickle.dump(l_0_ret, pk)

    with open('linf.pk', 'wb') as pk:
        pickle.dump(l_inf_ret, pk)
