import keras

from keras.layers import Input, Dense, Lambda, GRU, TimeDistributed
from keras.models import Model
import keras.backend as K

def make_vae_suite(batch_shape, latent_size):
    batch_size, seq_len, input_size = batch_shape

    # sampling function used by sampling layer
    def _sampling(args):
        sample_mean, sample_log_std = args
        epsilon = K.random_normal(shape=(batch_size, latent_size))
        return sample_mean + K.exp(sample_log_std) * epsilon

    # a function used in decoder input
    # to convert training data into RNN input form
    def _convert(data):
        data = K.concatenate([K.zeros(shape=(batch_size, 1, input_size)), data], axis=1)
        data = K.slice(data, [0, 0, 0], [batch_size, seq_len, input_size])
        return data

    # encoder
    encode_in = Input(batch_shape=batch_shape, name='encoder_in')
    hidden_state = GRU(latent_size)(encode_in)  # encoder GRU
    mean = Dense(latent_size)(hidden_state)
    log_std = Dense(latent_size)(hidden_state)

    # sampling layer
    # sample out z from the given mean and log_std
    z = Lambda(_sampling, name='sampling_layer')([mean, log_std])

    # decoder
    # instantiate layers in decoder
    convert_layer = Lambda(_convert, name='convert_decoder_input')
    decode_gru = GRU(latent_size, return_sequences=True, return_state=True)  # decoder GRU
    output_layer = TimeDistributed(Dense(input_size, activation='sigmoid'))


    # decoder for training (part of VAE)
    decode_in_train = convert_layer(encode_in)
    decode_out_train, _ = decode_gru(decode_in_train, initial_state=z)
    decode_out_train = output_layer(decode_out_train)

    ### end-to-end VAE ###
    vae = Model(encode_in, decode_out_train, name='VAE')

    # add VAE loss
    reconstruction_loss = K.mean(K.binary_crossentropy(K.reshape(encode_in, shape=(batch_size, -1)),\
            K.reshape(decode_out_train, shape=(batch_size, -1)))) * input_size * seq_len
    kl_loss = -0.5 * K.mean(1 + log_std - K.square(mean) - K.exp(log_std))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    ### ============== ###

    # encoder for inference (stand-alone encoder)
    encoder = Model(encode_in, [mean, log_std], name='encoder')

    # decoder for inference (stand-alone decoder)
    init_state = Input(batch_shape=(batch_size, latent_size), name='decoder_initial_state')
    decode_in = Input(batch_shape=batch_shape, name='decoder_in')

    decode_in_infer = convert_layer(decode_in)
    decode_out_infer, _ = decode_gru(decode_in_infer, initial_state=init_state)
    decode_out_infer = output_layer(decode_out_infer)

    decoder = Model([decode_in, init_state], decode_out_infer, name='decoder')

    return vae, encoder, decoder
# def make_vae_suite(batch_shape, latent_size):
#     batch_size, seq_len, input_size = batch_shape
#
#     # instantiate all necessary components
#     encoder = make_encoder(batch_shape, latent_size)
#     sampling_layer = make_sampling_layer(batch_size, latent_size)
#     decoder = make_decoder(batch_shape, latent_size)
#
#     # build VAE
#     vae_in = Input(batch_shape=batch_shape, name='VAE_input')
#     distribution = encoder(vae_in)
#     z = sampling_layer(distribution)
#     vae_out = decoder([z, vae_in])
#     vae = Model(vae_in, vae_out)
#
#     # add VAE loss
#     reconstruction_loss = K.mean(K.binary_crossentropy(K.reshape(vae_in, shape=(batch_size, -1)),\
#             K.reshape(vae_out, shape=(batch_size, -1)))) * input_size * seq_len
#     kl_loss = -0.5 * K.mean(1 + log_std - K.square(mean) - K.exp(log_std))
#     vae_loss = reconstruction_loss + kl_loss
#     vae.add_loss(vae_loss)
#
#     return vae
