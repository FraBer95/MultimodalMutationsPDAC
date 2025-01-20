import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from Variational_AE import Model,Encoder, Decoder
import pandas as pd
import numpy as np
def encoding(data):
    hidden_dim = 400
    latent_dim = 100
    input_dim = data.shape[1]

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = Model(Encoder=encoder, Decoder=decoder)
    model.load_state_dict(torch.load('models/vae_model.pth'))
    model = model.to('cuda')
    model.eval()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Converti i dati in un array NumPy
    data_np = data_scaled.astype(np.float32)

    data_tensor = torch.tensor(data_np, dtype=torch.float32)

    batch_size = 64
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():


        z_all = []
        for data, in data_loader:
            data = data.to('cuda')
            _, _, _, z = model(data)
            z_all.append(z.cpu().numpy())
        z_all = np.concatenate(z_all)
    z_df = pd.DataFrame(z_all)
    return z_df


if __name__ == '__main__':

    cptac_path = r'/genomic_analysis/datasets/autoencoder_data/cptac_5kgenes.csv'
    cptac = pd.read_csv(cptac_path)
    cptac_id = cptac['case_id']
    cptac_data = cptac.iloc[:, 1:]
    encoded_data = encoding(cptac_data)
    encoded_data = pd.concat([cptac_id, encoded_data], axis=1)
    encoded_data.to_csv('./vae_CPTAC.csv', index=False)
    # tcga5k_path = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\VAE\TCGA_5kgenes.csv'
    # cptac = pd.read_csv(cptac_path)
    # cptac.rename(columns={'Unnamed: 0': 'transcript'}, inplace=True)
    #
    # cptac.set_index('transcript', inplace=True)
    # df_T = cptac.T
    # df_T.reset_index(inplace=True)
    #
    # df_T.rename(columns={'index': 'case_id'}, inplace=True)
    # df_T['case_id'] = df_T['case_id'].str[:9]
    # tcga5k = pd.read_csv(tcga5k_path)
    # tcga5k = tcga5k.iloc[:, 1:]
    #
    # df_cptac5k = df_T[tcga5k.columns]
    # df_cptac5k_id = pd.concat([df_T['case_id'], df_cptac5k], axis=1)
    # df_cptac5k_id.to_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\VAE\cptac_5kgenes.csv',
    #                  index=False)