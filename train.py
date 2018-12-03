import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from model import ModelStruct
import data_utils

def train(epochs, batch_size, latent_size, save=True, plot=False):
    seq_len = 28
    input_size = 28
    batch_shape = (batch_size, seq_len, input_size)

    # construct models
    model_struct = ModelStruct(batch_shape, latent_size)
    vae = model_struct.assemble_vae_train()
    encoder = model_struct.assemble_encoder_infer()
    decoder = model_struct.assemble_decoder_infer()

    # plot if specified
    if plot:
        plot_model(vae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # load data and fit it to batch size
    x_train, _, x_test, _ = data_utils.load_data()
    x_train = data_utils.fit_batch(x_train, batch_size)
    x_test = data_utils.fit_batch(x_test, batch_size)

    # display compile and fit model
    vae.summary()
    vae.compile(optimizer='adam')
    vae.fit(x_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(x_test, None))

    # serialize trained models
    if save:
        encoder.save('saved_models/encoder.h5')
        print('encoder saved in \'saved_models/encoder.h5\'')
        decoder.save('saved_models/decoder.h5')
        print('decoder saved in \'saved_models/decoder.h5\'') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training program from mnist VAE')
    parser.add_argument('--batch', '-b', type=int, default=128,
                        help='batch size for training (deault 128)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of training epochs to perform (default 10)')
    parser.add_argument('--latent', '-l', type=int, default=64,
                        help='number of dimensions in latent representation (default 32)')
    args = parser.parse_args()
    
    train(args.epochs, args.batch, args.latent)
