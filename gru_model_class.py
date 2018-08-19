import keras

from keras.layers import Input, Dense, Lambda, GRU, TimeDistributed
from keras.models import Model
import keras.backend as K

class ModelStruct:
    def __init__(self, batch_shape, latent_size):
        self.__check_inputs(batch_shape, latent_size)  # check inputs are valid

        self.batch_shape = batch_shape
        self.latent_size = latent_size
        self.batch_size, self.seq_len, self.input_size = batch_shape
        self.non_batch_shape = batch_shape[1:]  # shape that doesn't consider batch_size

        ### encoder components ###
        self.encode_gru = GRU(self.latent_size)  # encoder GRU
        self.mean_output = Dense(self.latent_size)  # output layer for mean
        self.var_output = Dense(self.latent_size)  # output layer for variance

        ### sampling layer component ###
        self.sampling_layer = Lambda(self.__sampling, name='sampling_layer')

        ### decoder components ###
        self.convert_layer = Lambda(self.__convert, name='convert_layer')
        self.decode_gru = GRU(self.latent_size, return_sequences=True, return_state=True)  # decoder GRU
        self.output_dense = Dense(self.input_size, activation='sigmoid')

    # return an end-to-end VAE for training
    def assemble_vae_train(self):
        encode_in = Input(batch_shape=self.batch_shape, name='vae_in')
        hidden_state = self.encode_gru(encode_in)
        mean = self.mean_output(hidden_state)
        log_std = self.var_output(hidden_state)
        z = self.sampling_layer([mean, log_std])
        decode_in = self.convert_layer(encode_in)
        decode_state, _ = self.decode_gru(decode_in, initial_state=z)
        decode_out = TimeDistributed(self.output_dense)(decode_state)
        vae = Model(encode_in, decode_out, name='VAE')

        # add VAE loss
        reconstruction_loss = K.mean(K.binary_crossentropy(K.reshape(encode_in, shape=(self.batch_size, -1)),\
                K.reshape(decode_out, shape=(self.batch_size, -1)))) * self.input_size * self.seq_len
        kl_loss = -0.5 * K.mean(1 + log_std - K.square(mean) - K.exp(log_std))
        vae_loss = reconstruction_loss + kl_loss
        vae.add_loss(vae_loss)

        return vae

    '''
    Returns an encoder for inference, the components of this encoder is the same
    as the ones that made up the vae.

    The encoder, when called on .predict() method, returns the mean of code that
    represents input data.
    '''
    def assemble_encoder_infer(self):
        encode_in = Input(shape=self.non_batch_shape, name='encoder_in')
        hidden_state = self.encode_gru(encode_in)
        mean = self.mean_output(hidden_state)
        return Model(encode_in, mean)

    '''
    Returns a decoder for inference, the components of this decoder is the same
    as the ones that made up the vae.

    The decoder's .predict() method takes 2 parameters 'decoder input' and
    'initial state' respectively.

    The decoder, when called on .predict() method, returns the mean of code that
    represents input data.
    '''
    def assemble_decoder_infer(self):
        init_state = Input(shape=(self.latent_size,), name='decoder_initial_state')
        decode_in = Input(shape=(1, self.input_size), name='decoder_in')
        decode_states, hidden_state = self.decode_gru(decode_in, initial_state=init_state)
        decode_out = TimeDistributed(self.output_dense)(decode_states)
        return Model([decode_in, init_state], [decode_out, hidden_state], name='decoder')

    # helper method used to check inputs are valid
    # throws corresponding exceptions when expectations are not met
    def __check_inputs(self, batch_shape, latent_size):
        batch_shape_type = type(batch_shape)
        batch_shape_len = len(batch_shape)
        latent_size_type = type(latent_size)
        if batch_shape_type != tuple:
            raise TypeError('expect "batch_shape" to be type tuple, instead got {}'.format(batch_shape_type))
        elif batch_shape_len != 3:
            raise ValueError('expect "batch_shape" to have length == 3, instead got {}'.format(batch_shape_len))
        elif not all(i > 0 for i in batch_shape):
            raise ValueError('all elements in batch_shape must be greater than 0, instead got {}'.format(batch_shape))
        elif latent_size_type != int:
            raise TypeError('expect "latent_size" to be type int, instead got {}'.format(latent_size_type))
        elif latent_size <= 0:
            raise ValueError('expect "latent_size" to be greater than 0, instead got {}'.format(latent_size))

    # sampling function used by sampling layer
    def __sampling(self, args):
        sample_mean, sample_log_std = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_size))
        return sample_mean + K.exp(sample_log_std) * epsilon

    # a function used in decoder input
    # to convert training data into RNN input form
    def __convert(self, data):
        data = K.concatenate([K.zeros(shape=(self.batch_size, 1, self.input_size)), data], axis=1)
        data = K.slice(data, start=(0, 0, 0), size=self.batch_shape)
        return data
