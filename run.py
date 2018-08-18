import numpy as np
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
from gru_model_class import ModelStruct

batch_size = 128
seq_len = 28
input_size = 28
latent_size = 64

batch_shape = (batch_size, seq_len, input_size)

model_struct = ModelStruct(batch_shape, latent_size)
vae = model_struct.assemble_vae_train()
encoder = model_struct.assemble_encoder_infer()
decoder = model_struct.assemble_decoder_infer()

# plot_model(vae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# print('done plotting')

# load data
def load_data():
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_train = x_train / 255
        x_test, y_test = f['x_test'], f['y_test']
        x_test = x_test / 255
    return x_train, y_train, x_test, y_test

def fit_batch(data, batch):
    data_points = data.shape[0]
    remainder = data_points % batch
    return data[:data_points - remainder]

x_train, _, x_test, _ = load_data()
x_train = fit_batch(x_train, batch_size)
x_test = fit_batch(x_test, batch_size)

vae.summary()

vae.compile(optimizer='adam')
vae.fit(x_train, batch_size=batch_size, epochs=5, shuffle=True, validation_data=(x_test, None))

# reconstructed = vae.predict(x_test, batch_size=batch_size)

state = encoder.predict(x_test[0].reshape(1, 28, 28))
out = np.zeros((1, 1, 28), dtype=np.float32)  # initial "start" vector
predicted = []
for _ in range(28):
    out, state = decoder.predict([out, state])
    predicted.append(out.reshape(-1,))
predicted = np.stack(predicted, axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(x_test[0], cmap='gray')
ax1.set_axis_off()

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(predicted, cmap='gray')
ax2.set_axis_off()

plt.show()

# show results
# n = 5

# encoded_means, _ = encoder.predict(test_imgs)
# decoded_imgs_means = decoder.predict(encoded_means).reshape(-1, 28, 28)
# decoded_imgs_noise = vae.predict(test_imgs).reshape(-1, 28, 28)
# test_imgs = x_test[0: n]
# recon_imgs = reconstructed[0: n]
# fig = plt.figure()
# for i in range(1, n + 1):
#     # display original
#     ax = fig.add_subplot(2, n, i)
#     ax.imshow(test_imgs[i - 1], cmap='gray')
#     ax.set_axis_off()
#
#     # display mean reconstruction
#     ax = fig.add_subplot(2, n, i + n)
#     ax.imshow(recon_imgs[i - 1], cmap='gray')
#     ax.set_axis_off()
#
#     # display noisy reconstruction
#     # ax = fig.add_subplot(3, n, i + 2 * n)
#     # plt.imshow(decoded_imgs_noise[i - 1], cmap='gray')
#     # ax.set_axis_off()
#
# plt.show()
