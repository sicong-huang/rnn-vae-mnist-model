import numpy as np
from model import ModelStruct

batch_size = 2
seq_len = 28
input_size = 28
latent_size = 64

batch_shape = (batch_size, seq_len, input_size)

data = np.random.rand(batch_size, seq_len, input_size)

ms = ModelStruct(batch_shape, latent_size)
encoder = ms.make()

output = encoder.predict(data, batch_size=batch_size)
print(output)
