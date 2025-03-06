import os.path
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from autoencoder import build_autoencoder
import pandas as pd
import matplotlib.pyplot as plt
import keras
from utils import *


keras.utils.set_random_seed(42)


if __name__ == '__main__':
    data_path = '../datasets/autoencoder_data/raw_data/TCGA_5kgenes.csv'
    data_path_test = '../datasets/autoencoder_data/raw_data/cptac_5kgenes.csv'
    #dest_dir = r'../datasets/autoencoder_data/encoded_data/TCGA'
    #os.makedirs(dest_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    test =  pd.read_csv(data_path_test)

    data, sample_ids = prepare_data(data)
    data_norm = normalize_data(data)

    data_test, sample_ids_test = prepare_data(test)
    data_norm_test = normalize_data(data_test)

    train_data, val_data = train_test_split(data_norm, test_size=0.2, random_state=42)

    hidden_dims = [64, 128, 256]
    for dim in hidden_dims:
        print("Training AE with latent dimension ", dim)

        encoder, autoencoder = build_autoencoder(data.shape[1], hidden_dim=dim)
        autoencoder.compile(optimizer='adam', loss='mse')
        epochs = 500
        ckpt_dir = os.path.join('./models_new', str(dim))
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'cp.ckpt')

        es_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        ckpt_callback = ModelCheckpoint(filepath=ckpt_path , monitor='val_loss', save_best_only=True, save_weights_only=True)

        training = autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_data=(val_data, val_data),
            callbacks=[es_callback, ckpt_callback],
        )

        loss = training.history['loss']
        val_loss = training.history['val_loss']
        epochs = range(1, len(loss)+1)

        test_pred = autoencoder.predict(data_norm_test)
        mse = np.mean(np.power(data_norm_test - test_pred, 2), axis=1)

        # Se vuoi ottenere una media complessiva dell'MSE su tutto il dataset:
        mse_mean = np.mean(mse)

        print(f"Average MSE on {dim} on test data: {mse_mean}")


        # plt.figure()
        # plt.plot(epochs, loss,  label='Training loss')
        # plt.plot(epochs, val_loss,  label='Validation loss')
        # plt.title('Training and validation loss, z-score')
        # plt.legend()
        # plt.savefig(os.path.join(ckpt_dir, 'loss.png'))

        #encoded_data = encoder.predict(data_norm)

        #df_encoded = pd.DataFrame(encoded_data)
        #df_merged = pd.concat([sample_ids, df_encoded], axis=1)
        #df_merged.to_csv(os.path.join(dest_dir, f"TCGA_AE_5kgenes_{dim}.csv"), index=False)

        print("Finish")