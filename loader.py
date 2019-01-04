import pandas
import numpy


def convertSeriesToMatrix(vectorSeries, sequence_length):
    """define a function to convert a vector of time series into a 2D matrix"""
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix


class Instance(object):
    def __init__(self, _X_train, _y_train, _X_test, _y_test):
        self.X_train = _X_train
        self.y_train = _y_train
        self.X_test = _X_test
        self.y_test = _y_test


class Loader(object):
    def __init__(self, _data: str):
        self.raw_data = _data

    def load(self):
        df_raw = pandas.read_csv(self.raw_data, header=None)
        df_raw_array = df_raw.values
        list_daily_load = [df_raw_array[i, 5] / 100000 for i in range(0, len(df_raw)) if i % 24 != 0]
        sequence_length = 25
        matrix_load = numpy.array(convertSeriesToMatrix(list_daily_load, sequence_length))
        train_row = int(round(0.9 * matrix_load.shape[0]))
        train_set = matrix_load[:train_row:]
        X_train = train_set[:, :-1]
        y_train = train_set[:, -1]
        X_test = matrix_load[train_row:, :-1]
        y_test = matrix_load[train_row:, -1]
        X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return Instance(X_train, y_train, X_test, y_test)

