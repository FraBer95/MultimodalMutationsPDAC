import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def multimodal_preds(gen_pred_path, clam_pred_path, f_extractor): #file format = model_xx_GEne.csv

    for gene in os.listdir(gen_pred_path):
        #token_gene = gene.split('_')[-1]
        for clam in os.listdir(clam_pred_path):

            clam_size = clam.split('_')[0]
            gen_path = os.path.join(gen_pred_path, gene)
            clam_path = os.path.join(clam_pred_path, clam)

            token_clam = clam.split('_')[-1].split('.')[0]

            if gene == token_clam:
                if AE:
                    for model in os.listdir(gen_path):
                        csv_path = os.path.join(gen_path, model)
                        df_gen = pd.read_csv(csv_path)
                        df_clam = pd.read_csv(clam_path)

                        df_gen.rename(columns={'Y':'Y_g', 'Y_hat':'Y_hat_g'}, inplace=True)
                        df_clam.rename(columns={ 'Y_hat':'Y_hat_c'}, inplace=True)
                        merged_df = pd.merge(df_gen, df_clam, how='inner', on='case_id')

                        merged_df['mean_p0'] = merged_df[['p0', 'p_0']].mean(axis=1)
                        merged_df['mean_p1'] = merged_df[['p1', 'p_1']].mean(axis=1)
                        merged_df['Y_hat_multi'] = merged_df.apply(lambda row: 1 if row['mean_p1'] > row['mean_p0'] else 0, axis=1)

                        dest_dir = os.path.join('./multimodal_preds_new', gene)
                        os.makedirs(dest_dir, exist_ok=True)

                        models = f"{os.path.split(csv_path)[-1].split('.')[0]}_{f_extractor}_{clam_size}"
                        merged_df.to_csv(os.path.join(dest_dir, f"{models}.csv"), index=False)

                else:
                    for model in os.listdir(gen_path):
                        csv_path = os.path.join(gen_path, model)
                        df_gen = pd.read_csv(csv_path)
                        df_clam = pd.read_csv(clam_path)

                        df_gen.rename(columns={'Y': 'Y_g', 'Y_hat': 'Y_hat_g'}, inplace=True)
                        df_clam.rename(columns={'Y_hat': 'Y_hat_c'}, inplace=True)
                        merged_df = pd.merge(df_gen, df_clam, how='inner', on='case_id')

                        merged_df['mean_p0'] = merged_df[['p0', 'p_0']].mean(axis=1)
                        merged_df['mean_p1'] = merged_df[['p1', 'p_1']].mean(axis=1)
                        merged_df['Y_hat_multi'] = merged_df.apply(
                            lambda row: 1 if row['mean_p1'] > row['mean_p0'] else 0, axis=1)

                        dest_dir = os.path.join('./multimodal_preds_new', gene)
                        os.makedirs(dest_dir, exist_ok=True)

                        models = f"{os.path.split(csv_path)[-1].split('.')[0]}_{f_extractor}_{clam_size}"
                        merged_df.to_csv(os.path.join(dest_dir, f"{models}.csv"), index=False)





if __name__ == '__main__':
    AE = False

    f_extractors = ['Resnet', 'UNI', 'CONCH']
    path_to_clam = './probs_data/'

    if AE:
        print('Loading data with AE features...')
        gen_path = './probs_data/AE'

        for f_extractor in f_extractors:
            clam_path = os.path.join(path_to_clam, f_extractor)
            multimodal_preds(gen_path, clam_path, f_extractor)

    else:
        print('Loading data with DEG features...')
        gen_path = './probs_data/DEG'
        for f_extractor in f_extractors:
            clam_path = os.path.join(path_to_clam, f_extractor)
            multimodal_preds(gen_path, clam_path, f_extractor)
