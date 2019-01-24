import numpy
from keras import backend as K
from typing import List
import powerData.deepforge4.getPowerData
from keras.models import load_model, Model


def l_inf_attack(model, x, step_length, reverse: int, gradient, const_list: List[float]):
    ret = []
    adv_x = x.copy()
    get_grad_values = K.function([model.input], gradient)
    j = 0
    step = 1
    while True:
        grad_values = get_grad_values([[adv_x]])[0][0]
        grad_signs = numpy.sign(grad_values)  # get sign
        noise = grad_signs * step_length  # noise
        adv_x += noise * reverse

        if step == const_list[j]:
            tmp = adv_x.copy()
            ret.append(tmp)
            j += 1
            if j == len(const_list):
                break
        step += 1
    return ret


if __name__ == '__main__':
    dataobj = powerData.deepforge4.getPowerData.GetPowerData()
    data = dataobj.execute()
    _, _, test_x, test_y = data
    m = load_model('powerData/lstm150fc500.h5')
    grad = K.gradients(m.output, m.input)
    adv_test_x = l_inf_attack(m, test_x[0], 1e-3, -1, grad, [t for t in range(1, 20, 2)])
    print(len(adv_test_x))
