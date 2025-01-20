import os
import shap
import matplotlib.pyplot as plt
import keras
from genomic_analysis.AE.autoencoder import build_autoencoder
from genomic_analysis.AE.utils import *

if __name__ == '__main__':
    cptac_data = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets/autoencoder_data/raw_data/cptac_5kgenes.csv')
    #dest_dir = r'../datasets/autoencoder_data/encoded_data/CPTAC'
    os.makedirs(os.path.join('./', "XAI_explanations"), exist_ok=True)
    encoded_data = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\autoencoder_data\encoded_data\CPTAC\CPTAC_AE_5kgenes_128.csv')
    data, sample_ids = prepare_data(cptac_data)
    #data_norm = normalize_data(data)
    predictions = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\logs\AE\128\KRAS\RF.csv')

    hidden_dims = [128]
    for dim in hidden_dims:
        ckpt_dir = os.path.join('../models', str(dim))
        ckpt_path = os.path.join(ckpt_dir, 'cp.ckpt')
        encoder, autoencoder = build_autoencoder(data.shape[1], hidden_dim=dim)
        autoencoder.load_weights(ckpt_path).expect_partial()

        explainer = shap.KernelExplainer(predictions['Y_hat'], encoded_data.iloc[:,1:])
        shap_values_encoded = explainer.shap_values(encoded_data)
        shap.summary_plot(shap_values_encoded, encoded_data)
        print("Done!")

