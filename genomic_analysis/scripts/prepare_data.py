import pandas as pd
import os


"""Data preparation scripts for genomic analysis. The output data must be csv with name format '*_GENE.csv'
Each csv file should have the following columns "case_id, [gene expressions], label"
the inputfolder for training data preparation should be in format ".../datasets/classification_datasets/source_type"
"""

def prepare_data_DEG(gen_path, df_label, gene, dest_path, tcga):

    if tcga:
        print(f"Processing TCGA with gene: {gene}")
        df_gene = pd.read_csv(gen_path)
        df_label = df_label[['case_id', gene]]


        df_label.drop_duplicates(subset=['case_id'], inplace=True)
        df_label.rename(columns={gene: 'label'}, inplace=True)

        df_gene.rename(columns={'Unnamed: 0': 'transcript'}, inplace=True)

        df_gene.set_index('transcript', inplace=True)
        df_T = df_gene.T
        df_T.reset_index(inplace=True)
        df_T.rename(columns={'index': 'case_id'}, inplace=True)
        df_T['case_id'] = df_T['case_id'].str[:12]

        df_new = pd.merge(df_T, df_label, how='inner', on='case_id')

        final_dest = os.path.join(dest_path, f"DEG_{gene}.csv")
        print("Saving processed data to {}".format(final_dest))
        df_new.to_csv(final_dest, index=False)

    else:
        print("Processing CPTAC with gene: {gene}")

        df_gene = pd.read_csv(gen_path)
        df_label = df_label[['case_id', gene]]

        df_label.drop_duplicates(subset=['case_id'], inplace=True)
        df_label.rename(columns={gene: 'label'}, inplace=True)

        df_gene.rename(columns={'Unnamed: 0': 'transcript'}, inplace=True)
        df_gene.set_index('transcript', inplace=True)
        df_T = df_gene.T
        df_T.reset_index(inplace=True)
        df_T.rename(columns={'index': 'case_id'}, inplace=True)
        df_T['case_id'] = df_T['case_id'].str[:9]

        df_new = pd.merge(df_T, df_label, how='inner', on='case_id')

        final_dest = os.path.join(dest_path, f"DEG_{gene}.csv")
        print("Saving processed data to {}".format(final_dest))
        df_new.to_csv(final_dest, index=False)


def prepare_data_AE(gen_path, df_label, gene, dest_path, tcga):

    if tcga:
        print(f"Processing TCGA with gene: {gene}")
        df_gene = pd.read_csv(gen_path)
        df_label = df_label[['case_id', gene]]

        df_gene['case_id'] = df_gene['case_id'].str[:12]
        df_label.drop_duplicates(subset=['case_id'], inplace=True)
        df_label.rename(columns={gene: 'label'}, inplace=True)

        df_new = pd.merge(df_gene, df_label, how='inner', on='case_id')

        final_dest = os.path.join(dest_path, f"AE_{gene}.csv")
        print("Saving processed data to {}".format(final_dest))
        df_new.to_csv(final_dest, index=False)

    else:
        print("Processing CPTAC with gene: {gene}")

        df_gene = pd.read_csv(gen_path)
        df_label = df_label[['case_id', gene]]
        df_gene['case_id'] = df_gene['case_id'].str[:12]

        df_label.drop_duplicates(subset=['case_id'], inplace=True)
        df_label.rename(columns={gene: 'label'}, inplace=True)

        df_new = pd.merge(df_gene, df_label, how='inner', on='case_id')

        final_dest = os.path.join(dest_path, f"AE_{gene}.csv")
        print("Saving processed data to {}".format(final_dest))
        df_new.to_csv(final_dest, index=False)




if __name__ == '__main__':
    tcga = True
    #general_path = '../datasets/autoencoder_data/encoded_data'
    general_path = '../datasets/DEGDataTCGA'
    source = os.path.split(general_path)[-1]

    genes = ['CDKN2A', 'KRAS', 'SMAD4', 'TP53', 'TTN']

    if source == 'DEGDataTCGA':
        if tcga:
            print("Processing TCGA DEG dataset...")
            dest_path = '../datasets/classification_dataset/DEG'
            os.makedirs(dest_path, exist_ok=True)
            label_df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\full_datasets\tcga_filtered_mut.csv')

            for filename in os.listdir(general_path):
                gene = filename.split('_')[1].split('.')[0]
                if gene in genes:
                    full_gene_path = os.path.join(general_path, filename) #path completo al file
                    prepare_data_DEG(full_gene_path, label_df, gene, dest_path, tcga)
        else:
            print("Processing CPTAC DEG dataset (for test)...")
            dest_path = '../datasets/test_set/DEG'
            os.makedirs(dest_path, exist_ok=True)
            gene_path = r'../datasets/test_set/full_matrix/cptac_matrix.csv'
            label_df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\full_datasets\cptac_filtered_mut.csv')

            for filename in os.listdir(general_path):
                gene = filename.split('_')[1].split('.')[0]
                if gene in genes:
                    prepare_data_DEG(gene_path, label_df, gene, dest_path, tcga)

    elif source == 'encoded_data':
        if tcga:
            print("Processing TCGA Autoencoder dataset...")
            dest_path = '../datasets/classification_dataset/AE'
            os.makedirs(dest_path, exist_ok=True)
            label_df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\full_datasets\tcga_filtered_mut.csv')
            tcga_path = os.path.join(general_path, 'TCGA')

            for filename in os.listdir(tcga_path):
                size = filename.split('_')[-1].split('.')[0]
                dest_path_size = os.path.join(dest_path, size)
                os.makedirs(dest_path_size, exist_ok=True)
                full_gene_path = os.path.join(tcga_path, filename)  # path completo al file

                for gene in genes:
                    prepare_data_AE(full_gene_path, label_df, gene, dest_path_size, tcga)

        else:
            print("Processing CPTAC Autoencoder dataset (for test)...")
            dest_path = '../datasets/test_set/AE'
            os.makedirs(dest_path, exist_ok=True)

            label_df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\full_datasets\cptac_filtered_mut.csv')
            cptac_path = os.path.join(general_path, 'CPTAC')

            for filename in os.listdir(cptac_path):
                size = filename.split('_')[-1].split('.')[0]
                dest_path_size = os.path.join(dest_path, size)
                os.makedirs(dest_path_size, exist_ok=True)

                full_gene_path = os.path.join(cptac_path, filename)
                for gene in genes:
                    prepare_data_AE(full_gene_path, label_df, gene, dest_path_size, tcga)


    else:
        raise Exception(f"Data source not supported! Source specified: {source}")

