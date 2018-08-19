import numpy as np

# load mnist data
def load_data():
    with np.load('data/mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_train = x_train / 255
        x_test, y_test = f['x_test'], f['y_test']
        x_test = x_test / 255
    return x_train, y_train, x_test, y_test

# a function to trim the first axis of data to fit the given batch size
def fit_batch(data, batch):
    data_points = data.shape[0]
    remainder = data_points % batch
    return data[:data_points - remainder]
