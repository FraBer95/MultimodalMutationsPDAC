from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import pandas as pd
import torch
from Variational_AE import Encoder, Decoder, Model
import matplotlib.pyplot as plt

if __name__=='__main__':
    data_path = '../datasets/autoencoder_data/raw_data/TCGA_5kgenes.csv'
    data = pd.read_csv(data_path)

    sample_id = data.iloc[:,:1]

    sample_id = sample_id.applymap(lambda x: x.replace('.', '-') if isinstance(x, str) else x)
    sample_id = sample_id.rename(columns={'Unnamed: 0': 'case_id'})
    sample_id = sample_id['case_id'].str[:12]
    data = data.iloc[:, 1:]
    scaler = StandardScaler(with_mean=True, with_std=True)
    data_scaled = scaler.fit_transform(data)

    # Converti i dati in un array NumPy
    data_np = data_scaled.astype(np.float32)
    train_data, val_data = train_test_split(data_np, test_size=0.2, random_state=42)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)

    batch_size = 64
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = data.shape[1]
    hidden_dim = 400
    latent_dim = 100
    learning_rate = 1e-4
    num_epochs = 500

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to('cuda')


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tr_loss = []
    vd_loss = []

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to('cuda')
            optimizer.zero_grad()
            x_hat, mu, logvar, _ = model(data)
            loss = model.loss_function(x_hat, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Running loss on scripts {train_loss/len(train_loader.dataset)}")


        model.eval()
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to('cuda')
                x_hat, mu, logvar, _ = model(data)
                loss = model.loss_function(x_hat, data, mu, logvar)
                val_loss += loss.item()
            print(f"Running loss on validation {val_loss/len(val_loader.dataset)}")

        model.train()
        tr_loss.append(train_loss/len(train_loader.dataset))
        vd_loss.append(val_loss/len(val_loader.dataset))

    # Plot delle curve di loss
    plt.figure(figsize=(10, 5))
    plt.plot(tr_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    #plt.yticks([1000, 1500, 2000, 2500, 3000, 3500])
    plt.yticks()
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    model.eval()
    total_mse_loss = 0
    total_mae_loss = 0
    num_batches = 0
    with torch.no_grad():
        for data, in val_loader:
            data = data.to('cuda')
            x_hat, mu, logvar, _ = model(data)
            total_mse_loss += nn.MSELoss(reduction='sum')(x_hat, data).item()
            total_mae_loss += nn.L1Loss(reduction='sum')(x_hat, data).item()
            num_batches += 1

    mean_mse = total_mse_loss / len(val_loader.dataset)
    mean_mae = total_mae_loss / len(val_loader.dataset)

    print(f"Final Mean MSE Error: {mean_mse}")
    print(f"Final Mean MAE Error: {mean_mae}")

    #torch.save(model.state_dict(), './vae_model_2.pth')

    # model.eval()
    # with torch.no_grad():
    #     z_all = []
    #     for data, in data_loader:
    #         data = data.to('cuda')
    #         _, _, _, z = model(data)
    #         z_all.append(z.cpu().numpy())
    #     z_all = np.concatenate(z_all)
    #     z_df = pd.DataFrame(z_all)
    #     latent_df = pd.concat([pd.DataFrame(sample_id), z_df], axis=1)
    #     #latent_df.to_csv(r'./vae_tcga.csv', index=False)
    print("Finish")
