import numpy as np
import pandas as pd
import os
import re
"""Leggo i csv con le probabilità, per ogni fold, per ogni task. Faccio la media di ogni fold e poi medio 
eventuali probabilità associate allo stesso paziente"""

def ensamble_clam(path):

    gene = path.split('_')[-2]
    model = path.split('_')[-1]
    df_dict = {}

    if gene in ['KRAS', 'SMAD4', 'TP53', 'TTN', 'CDKN2A']:
        pattern = re.compile(r'.*_\d+\.csv$')
        print(f"Processing predictions of {gene} ")

        for csv_file in os.listdir(path):
            if pattern.match(csv_file):

                df_path = os.path.join(path, csv_file)
                df = pd.read_csv(df_path)
                df_dict[csv_file] = df

        combined_df = pd.concat(df_dict.values(), ignore_index=True)

        combined_df['slide_id'] = combined_df['slide_id'].str[:-3]
        combined_df.rename(columns={'slide_id': 'case_id'}, inplace=True)

        mean_predictions = combined_df.groupby('case_id').agg({
            'p_0': 'mean',
            'p_1': 'mean',
            'Y': 'first',
            'Y_hat': 'mean'
        }).reset_index()

        dest_path = os.path.join('probs_data/Resnet')
        os.makedirs(dest_path, exist_ok=True)

        mean_predictions.to_csv(os.path.join(dest_path, f'{model}_{gene}.csv'), index=False)








if __name__ == '__main__':

    gen_path = r'E:\Users\Berloco\PycharmProjects\CLAM\eval_results_Resnet'

    for fold_name in os.listdir(gen_path):
        fold_path = os.path.join(gen_path, fold_name)
        ensamble_clam(fold_path)