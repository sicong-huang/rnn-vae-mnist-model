import numpy as np
import keras
# import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, GRU, TimeDistributed
from keras.models import Model
import keras.backend as K

batch_size = 128
input_size = 28
seq_len = 28
latent_size = 64

# define encoder
encode_in = Input(batch_shape=(batch_size, seq_len, input_size), name='encoder_in')
hidden_state = GRU(latent_size)(encode_in)
mean = Dense(latent_size)(hidden_state)
log_std = Dense(latent_size)(hidden_state)

encoder_train = Model(encode_in, [mean, log_std], name='encoder')

# define sampling layer
def sampling(args):
    sample_mean, sample_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_size))
    return sample_mean + K.exp(sample_log_std) * epsilon

sampling_layer = Lambda(sampling, name='sampling_layer')

### define decoders
# decoder GRU
initial_state = Input(batch_shape=(batch_size, latent_size), name='decoder_initial_state')
decode_gru = GRU(latent_size, return_sequences=True, return_state=True)

# decoder for training
def convert(data):
    data = K.concatenate([K.zeros(shape=(batch_size, 1, input_size)), data], axis=1)
    data = K.slice(data, [0, 0, 0], [batch_size, seq_len, input_size])
    return data

decode_in_train = Input(batch_shape=(batch_size, seq_len, input_size), name='decoder_in')
decode_in = Lambda(convert, name='convert_decoder_input')(decode_in_train)
decode_out_train, _ = decode_gru(decode_in, initial_state=initial_state)
decode_out_train = TimeDistributed(Dense(input_size))(decode_out_train)

decoder_train = Model([initial_state, decode_in_train], decode_out_train, name='decoder_train')

# end-to-end autoencoder
vae_in = Input(batch_shape=(batch_size, seq_len, input_size), name='VAE_input')
distribution = encoder_train(vae_in)
z = sampling_layer(distribution)
vae_out = decoder_train([z, vae_in])

vae = Model(vae_in, vae_out)

# define loss
reconstruction_loss = K.mean(K.binary_crossentropy(K.reshape(vae_in, shape=(batch_size, -1)), K.reshape(vae_out, shape=(batch_size, -1)))) * input_size * seq_len
kl_loss = -0.5 * K.mean(1 + distribution[1] - K.square(distribution[0]) - K.exp(distribution[1]))
vae_loss = reconstruction_loss + kl_loss

# define variance metric function
# def variance(var):
#     def get_variance(y_true, y_pred):
#         return K.mean(distribution[1])
#     return get_variance

# def vae_loss(x, x_decoded_mean):
#     xent_loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
#     kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
#     return xent_loss + kl_loss

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

# model summary
vae.summary()

# train model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.fit(x_train, batch_size=batch_size, epochs=3, shuffle=True)

# evaluate model
# print('evaluating...')
# test_loss = vae.evaluate(x_test, batch_size=batch_size)
# print('evaluation_loss = {:.6f}'.format(test_loss))
#
# # show results
# n = 5
# test_imgs = x_test[10: 10 + n]
# encoded_means, _ = encoder.predict(test_imgs)
# decoded_imgs_means = decoder.predict(encoded_means).reshape(-1, 28, 28)
# decoded_imgs_noise = vae.predict(test_imgs).reshape(-1, 28, 28)
#
# fig = plt.figure()
# for i in range(1, n + 1):
#     # display original
#     ax = fig.add_subplot(3, n, i)
#     ax.imshow(test_imgs[i - 1].reshape(28, 28), cmap='gray')
#     ax.set_axis_off()
#
#     # display mean reconstruction
#     ax = fig.add_subplot(3, n, i + n)
#     ax.imshow(decoded_imgs_means[i - 1], cmap='gray')
#     ax.set_axis_off()
#
#     # display noisy reconstruction
#     ax = fig.add_subplot(3, n, i + 2 * n)
#     plt.imshow(decoded_imgs_noise[i - 1], cmap='gray')
#     ax.set_axis_off()
#
# plt.show()
