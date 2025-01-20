import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.models import Model
import tensorflow as tf


def prepare_data(df):

    sample_id = df.iloc[:, :1]

    sample_id = sample_id.applymap(lambda x: x.replace('.', '-') if isinstance(x, str) else x)
    sample_id = sample_id.rename(columns={'Unnamed: 0': 'case_id'})
    sample_id = sample_id['case_id'].str[:12]
    elab_df = df.iloc[:, 1:]

    return elab_df, sample_id

def normalize_data(df):

    scaler = RobustScaler(quantile_range=(1.5, 98.5))
    data_scaled = scaler.fit_transform(df)
    data_np = data_scaled.astype(np.float32)

    return data_np


def combine_models(encoder, classifier):
    for layer in encoder.layers:
        layer.trainable = False

    encoded = encoder.output
    combined_output = classifier(encoded)
    combined_model = Model(inputs=encoder.input, outputs=combined_output)
    return combined_model


def calculate_jacobian(encoder_model, input_data):
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        encoded_output = encoder_model(input_data)
    jacobian = tape.batch_jacobian(encoded_output, input_data)
    return jacobian
