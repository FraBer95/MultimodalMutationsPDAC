import keras
from keras import layers



def build_autoencoder(data_dim, hidden_dim=128):

    input_data = keras.Input(shape=(data_dim,))
    encoded = layers.Dense(hidden_dim*4, activation='relu')(input_data)
    #encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dense(hidden_dim*2, activation='relu')(encoded)
    #encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dense(hidden_dim, activation='relu')(encoded)

    decoded = layers.Dense(hidden_dim*2, activation='relu')(encoded)
    #decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(hidden_dim*4, activation='relu')(decoded)
    #decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(data_dim)(decoded)

    encoder = keras.Model(input_data, encoded)
    autoencoder = keras.Model(input_data, decoded)

    return encoder, autoencoder

