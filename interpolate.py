import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
import data_utils

encoder = load_model('saved_models/encoder.h5')
decoder = load_model('saved_models/decoder.h5')

def encode(img):
    return encoder.predict(img.reshape(1, 28, 28)).reshape(-1)

def decode(code):
    code = code.reshape(1, -1)
    out = np.zeros((1, 1, 28), dtype=np.float32)  # initial "start" vector
    predicted = []
    for _ in range(28):
        out, code = decoder.predict([out, code])
        predicted.append(out.reshape(-1))
    return np.stack(predicted, axis=0)

# a np.linspace function on vectors
def linspace(start, end, N):
    step = (end - start) / (N - 1)
    return step * np.arange(N)[:, None] + start

def imshow(ax, img, title=None):
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    if title != None:
        ax.set_title(title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program to interpolate between mnist digits')
    parser.add_argument('--start', '-s', type=int, default=5)
    parser.add_argument('--end', '-e', type=int, default=110)
    args = parser.parse_args()

    interpolate_steps = 8
    _, _, x_test, y_test = data_utils.load_data()
    img_1 = x_test[args.start]
    img_2 = x_test[args.end]
    encoded_1 = encode(img_1)
    encoded_2 = encode(img_2)

    fig_orig, axes_orig = plt.subplots(1, 2, figsize=(4, 2))
    fig_orig.suptitle('origianl digits')
    imshow(axes_orig[0], img_1, 'start')
    imshow(axes_orig[1], img_2, 'end')

    fig_interp, axes_interp = plt.subplots(1, interpolate_steps, figsize=(16, 2))
    fig_interp.suptitle('interpolated digits')
    interpolated = linspace(encoded_1, encoded_2, interpolate_steps)
    for code, ax in zip(interpolated, axes_interp):
        decoded = decode(code)
        imshow(ax, decoded)
    plt.show()
