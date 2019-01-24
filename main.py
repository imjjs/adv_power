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
    model = load_model('powerData/lstm150fc500.h5')
    # model.summary()
    # res_model = set_testing_dropout(model, .1)
    grad = K.gradients(model.output, model.input)
    thrs = [.1, .15, .2]
    for thr in thrs:

        with open('min_{pc}%_l0.pk'.format(pc=str(thr*100)), 'wb') as pk:
            pickle.dump(adv_test_x, pk)
    # result = model.predict(test_x)
    # adv_result = model.predict(adv_test_x)
    # adv_res_test_x = trainsform(adv_test_x)
    # adv_res_result = model.predict(adv_res_test_x)
    # res_test_x = trainsform(test_x)
    # res_result = model.predict(res_test_x)
    # rd = deviation(test_y, result)
    # adv_rd = deviation(test_y, adv_result)
    # adv_res_rd = deviation(test_y, adv_res_result)
    # res_rd = deviation(test_y, res_result)
    # mse = mean_squared_error(test_y, result)
    # adv_mse  = mean_squared_error(test_y, adv_result)
    # adv_res_mse = mean_squared_error(test_y, adv_res_result)
    # res_mse = mean_squared_error(test_y, res_result)
    # print(mse, adv_mse, adv_res_mse, res_mse)
    # print(sum(rd)/len(rd))
    # print(sum(adv_rd) / len(adv_rd))
    # print(sum(adv_res_rd) / len(adv_res_rd))
    # print(sum(res_rd)/ len(res_rd))
    # for idx in range(len(res_rd)):
    #     print(result[idx], adv_result[idx], res_result[idx])


    # exp = model.Experiment(ins)
    # raw_predictions = exp.predict_test()
    # predictions = [x[0] for x in raw_predictions]
    # #K.set_learning_phase(0)
    # X_advs = exp.attack()
    # X_advs = numpy.reshape(X_advs, (207, 24, 1))
    # exp.set_testing_dropout(.3)
    # raw_adv_predictions = exp.predict(X_advs)
    # adv_predictions = [x[0] for x in raw_adv_predictions]
    # ret = deviation(predictions, adv_predictions)
    # print(sum(ret)/len(ret))
