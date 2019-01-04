import loader
import model
import numpy
from keras import backend as K


def deviation(original, adv):
    assert(len(original) == len(adv))
    ret = []
    for idx in range(len(original)):
        rd = (adv[idx] - original[idx]) / original[idx]
        #print('original: ' + str(original[idx]) + ', adv: ' + str(adv[idx]) + ', deviation: ' + str(rd))
        ret.append(rd)
    return ret

if __name__ == '__main__':
    l = loader.Loader('data_load.csv')
    ins = l.load()
    exp = model.Experiment(ins)
    raw_predictions = exp.predict_test()
    predictions = [x[0] for x in raw_predictions]
    K.set_learning_phase(0)
    X_advs = exp.attack(-1)
    X_advs = numpy.reshape(X_advs, (207, 24, 1))
    print(X_advs)
    raw_adv_predictions = exp.predict(X_advs)
    adv_predictions = [x[0] for x in raw_adv_predictions]
    ret = deviation(predictions, adv_predictions)
    print(sum(ret)/len(ret))