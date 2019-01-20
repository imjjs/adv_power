import loader
import model
import numpy
import pickle
import powerData.deepforge4.getPowerData
from keras import backend as K
from keras.models import load_model, Model
from keras.layers.core import Lambda
from sklearn.metrics import mean_squared_error


def deviation(original, adv):
    assert(len(original) == len(adv))
    ret = []
    for idx in range(len(original)):
        rd = (adv[idx] - original[idx]) / original[idx]
        #print('original: ' + str(original[idx]) + ', adv: ' + str(adv[idx]) + ', deviation: ' + str(rd))
        ret.append(rd)
    return ret


def l0_attack(model, X_tst, reverse, grad, threshold):
    ret = []
    original_prediction = model.predict(X_tst)
    get_grad_values = K.function([model.input], grad)
    max_value = X_tst.max()
    min_value = X_tst.min()
    shift_value = 5.62481483
    for i in range(len(X_tst)):
        print('ins: ' + str(i))
        x = X_tst[i]
        x_adv = x.copy()
        cnt = 0
        cnt_failure = 0
        prediction = original_prediction[i][0]
        adv_prediction = prediction
        gradient = get_grad_values([numpy.array([x])])[0][0]
        abs_gradient = numpy.abs(gradient)
        while (adv_prediction - prediction) * reverse / (prediction + shift_value) <= threshold:
            argmax_num = numpy.nanargmax(abs_gradient)
            argmax_indices = numpy.unravel_index(argmax_num, abs_gradient.shape)
            abs_gradient[argmax_indices] = 0  # to find the next largest gradient pixel
            if reverse == 1:
                best_value = max_value if gradient[argmax_indices] > 0 else min_value
            else:
                best_value = min_value if gradient[argmax_indices] > 0 else max_value
            saved_value = x_adv[argmax_indices]
            flag_better = False
            for k in numpy.linspace(min_value, max_value, num=10):
                x_adv[argmax_indices] = k
                y_pred_tmp = model.predict(numpy.array([x_adv]))[0][0]
                if (y_pred_tmp - adv_prediction) * reverse > 0:
                    best_value = k
                    adv_prediction = y_pred_tmp
                    flag_better = True
    #
            print(cnt, gradient[argmax_indices], argmax_indices, best_value,
                  adv_prediction, flag_better)
            x_adv[argmax_indices] = best_value if flag_better else saved_value
            cnt = cnt + 1 if flag_better else cnt
        ret.append((x_adv, cnt))
    return numpy.array(ret)



def multiple_IterativeGSM(model, X_tst, step_length, steps, reverse, grad, ratio):
    x_copy = X_tst.copy()
    get_grad_values = K.function([model.input], grad)
    for step in range(steps):
        # stepLen = step_length * (x_copy.max_value - x_copy.min_value) / steps
        grad_values = get_grad_values([x_copy])[0]
        grad_signs = numpy.sign(grad_values)  # get sign
        noise = grad_signs * step_length  # noise
        x_copy = x_copy + noise * reverse
    # for i in range(len(x_copy)):
    #     for j in range(len(x_copy[0])):
    #         tmp = numpy.random.randint(0, 100)
    #         if tmp < 100 * ratio:
    #             x_copy[i][j] = X_tst[i][j]
    return x_copy

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


import numpy as np
def trainsform(original_test_x):
#    mu = original_test_x.mean()
#    sigma = original_test_x.std()
    
    ratio = .1
    ret = original_test_x.copy()
    for i in range(len(original_test_x)):
        for j in range(len(original_test_x[0])):
            tmp = numpy.random.randint(0, 100)
            if tmp < 100 * ratio:
                ret[i][j] = ret[i][j].mean()
#                if ret[i][j].mean()< mu:
#                    ret[i][j] = mu+ sigma
#                else:
#                    ret[i][j] = mu- sigma
    np.clip(ret,0,1)
    # z = numpy.zeros(original_test_x.shape)
    # tmp = numpy.random.randint(0, 100, size=original_test_x.shape)
    # ret = numpy.where(tmp < 100 * ratio, original_test_x, z)
    # ret = ret / (1-ratio)
    return ret

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
        adv_test_x = l0_attack(model, test_x, -1, grad, thr)
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
