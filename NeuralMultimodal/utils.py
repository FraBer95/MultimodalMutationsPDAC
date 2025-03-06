import re
import os
import pandas as pd


def check_align(img_df, rna_df, train):

    if train == True:
        if img_df['slide_id'].str[:12].equals(rna_df['case_id']):
            print("Dataframe allineati")
        else: raise Exception("Dataframe non allineati")

    else:
        df_pairs = pd.read_json(
            r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\TCIA CPTAC Pathology Portal.json')
        df_pairs = df_pairs[df_pairs.Specimen_Type == 'tumor_tissue']

        df_pairs = df_pairs[df_pairs['Specimen_ID'].isin(rna_df['new_case_id'])]
        new_img_df = img_df.set_index('slide_id').reindex(df_pairs['Slide_ID']).reset_index()
        new_rna = rna_df.set_index('new_case_id').reindex(df_pairs['Specimen_ID']).reset_index()
        return new_img_df, new_rna





def read_multi_sets(path, train):
    pattern = re.compile(r'.*_\d+\.csv')  # regex for Features_foldN.csv

    for files in os.listdir(path):  # load data

        if pattern.match(files):
            img_df = pd.read_csv(os.path.join(path, files))
        else:
            rna_df = pd.read_csv(os.path.join(path, files))


    check_align(img_df, rna_df, train)
    return img_df, rna_df