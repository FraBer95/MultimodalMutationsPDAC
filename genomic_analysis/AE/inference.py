import os

from autoencoder import build_autoencoder
import matplotlib.pyplot as plt
import keras
from utils import *

if __name__ == '__main__':
    cptac_data = pd.read_csv('../datasets/autoencoder_data/raw_data/cptac_5kgenes.csv')
    dest_dir = r'../datasets/autoencoder_data/encoded_data/CPTAC'
    #os.makedirs(dest_dir, exist_ok=True)

    data, sample_ids = prepare_data(cptac_data)
    data_norm = normalize_data(data)

    hidden_dims = [64, 128, 256]
    for dim in hidden_dims:
        ckpt_dir = os.path.join('./models', str(dim))
        ckpt_path = os.path.join(ckpt_dir, 'cp.ckpt')
        encoder, autoencoder = build_autoencoder(data.shape[1], hidden_dim=dim)
        autoencoder.load_weights(ckpt_path).expect_partial()

        encoded_data = encoder.predict(data_norm)


        df_encoded = pd.DataFrame(encoded_data)
        df_merged = pd.concat([sample_ids, df_encoded], axis=1)
        df_merged.to_csv(os.path.join(dest_dir, f"CPTAC_AE_5kgenes_{dim}.csv"), index=False)

