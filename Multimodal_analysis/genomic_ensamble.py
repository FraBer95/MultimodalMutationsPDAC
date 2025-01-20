import numpy as np
import pandas as pd
import os
import re
"""Leggo i csv con le probabilità, per ogni fold, per ogni task. Faccio la media di ogni fold e poi medio 
eventuali probabilità associate allo stesso paziente"""

def ensamble_gen(path, fold):

        #label_path = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\Multimodal_analysis\probs_data\CONCH'
        for csv_file in os.listdir(path):
            model = csv_file.split('_')[0]
            if csv_file.endswith('.csv'):

                path_df = os.path.join(path, csv_file)
                df = pd.read_csv(path_df)
                mean_predictions = df.groupby('case_id').agg({
                    'p0': 'mean',
                    'p1': 'mean',
                    'Y': 'first'
                }).reset_index()

                y_pred = np.argmax(mean_predictions[['p0', 'p1']].to_numpy(), axis=1)
                mean_predictions = pd.concat([mean_predictions, pd.DataFrame(y_pred, columns=['Y_hat'])], axis=1)
                dest_path = os.path.join(r'/genomic_analysis/logs/VAE/ensamble_predictions', model)
                os.makedirs(dest_path, exist_ok=True)
                mean_predictions.to_csv(os.path.join(dest_path, f'{model}_mean_KRAS.csv'), index=False)

                # for file in os.listdir(label_path):
                #     gen_file = path_df.split('\\')[-2]
                #     lab_file = file.split('_')[-1]
                #     lab_file = lab_file.split('.')[0]
                #     if gen_file == lab_file:
                #         label_df = os.path.join(label_path, file)
                #         label_df = pd.read_csv(label_df)
                #         merged = pd.merge(mean_predictions, label_df, on='case_id')
                # mean_predictions['Y'] = merged['Y']
                # mean_predictions.to_csv(os.path.join(dest_path, f'{csv_file[:-4]}_{fold}.csv'), index=False)








if __name__ == '__main__':

    gen_path = r'/genomic_analysis/logs/AE'

    for fold_name in os.listdir(gen_path):
        fold_path = os.path.join(gen_path, fold_name)
        ensamble_gen(fold_path, fold_name)