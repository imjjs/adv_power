import numpy
from keras import backend as K
from typing import List
import powerData.deepforge4.getPowerData
from keras.models import load_model, Model


def l0_attack(model, x, reverse: int, grad, const_list: List[int]):
    get_grad_values = K.function([model.input], grad)
    max_value = 1.0
    min_value = 0.0
    x_adv = x.copy()
    cnt = 0
    prediction = model.predict(numpy.array([x]))[0][0]
    adv_prediction = prediction
    gradient = get_grad_values([numpy.array([x])])[0][0]
    abs_gradient = numpy.abs(gradient)
    adv_input = []
    j = 0
    while True:
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
        if flag_better is True:
            x_adv[argmax_indices] = best_value
            cnt += 1
        else:
            x_adv[argmax_indices] = saved_value
        if cnt == const_list[j]:
            tmp = x_adv.copy()
            adv_input.append(tmp)
            #adv_result.append(adv_prediction)
            j += 1
            if j == len(const_list):
                break
    return adv_input


if __name__ == '__main__':
    dataobj = powerData.deepforge4.getPowerData.GetPowerData()
    data = dataobj.execute()
    _, _, test_x, test_y = data
    model = load_model('powerData/lstm150fc500.h5')
    grad = K.gradients(model.output, model.input)
    adv_test_x = l0_attack(model, test_x[0], -1, grad, [t for t in range(1, 20, 2)])
    print(list(range(1,20,2)))