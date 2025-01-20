import unittest
import os
import shutil
import pandas as pd
import re
from tqdm import tqdm

class MyTestCase(unittest.TestCase):
    def test_images_prep(self):

        origin = r'E:\Datasets\DP_Pancreas\TCGA\svs\*'
        dest = r'E:\Datasets\DP_Pancreas\Clam_data'
        if not os.path.exists(dest):
            os.makedirs(dest)

            # Scorri tutte le sottocartelle nella cartella di origine
        for root, dirs, files in os.walk(origin):
            # Scorri tutti i file nella sottocartella corrente
            for file in files:
                # Verifica se il file ha estensione .svs
                if file.endswith('.svs'):
                    # Costruisci il percorso completo del file di origine e di destinazione
                    origine_file = os.path.join(root, file)
                    destinazione_file = os.path.join(dest, file)
                    # Sposta il file nella cartella di destinazione
                    shutil.move(origine_file, destinazione_file)


    def test_class_reduction(self):
        df = pd.read_csv(r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\tcga_paad_histotype.csv")
        print(df["label"].value_counts())

        df.loc[df['label'] == 'Infiltrating duct carcinoma, NOS', 'label'] = 'Infiltrating'
        df.loc[df['label'] != 'Infiltrating', 'label'] = 'Other_type'

        print(df["label"].value_counts())
        df.to_csv(r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\tcga_paad_histotype_new.csv", index=False)



    def test_csv(self):
        df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\TCGA_pathes\process_list_autogen.csv', sep=';')
        df.to_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\TCGA_pathes\process_list_autogen_new.csv', sep=',', index=False)



    def test_prepare_CPTAC(self):

        clin_data = pd.read_csv(r'E:\Datasets\DP_Pancreas\CPTAC\CPTAC_PDA_v8\clinical_data.csv', sep=',')


        patients_data = clin_data[['patient_id', 'df_clinical.vital_status']]


        status_count = patients_data['df_clinical.vital_status'].value_counts()
        patients_data.drop_duplicates(inplace=True)

        slides = os.listdir(r'E:\Datasets\DP_Pancreas\CPTAC\CPTAC_PDA_v8\PDA')

        slides_df = pd.DataFrame(slides, columns=['slide_id'])


        slides_patient = slides_df['slide_id'].str.split('.').str[0]

        filtered_slide = slides_patient[slides_patient.str.contains('|'.join(patients_data))]
        filtered_slide = pd.DataFrame(filtered_slide, columns=['slide_id'])

        df_joined = []
        for index, row in clin_data.iterrows():
            case_id = row['patient_id']

            slide_ids = filtered_slide[filtered_slide['slide_id'].str.contains(case_id)]['slide_id'].str.split(',')

            for slide_id in slide_ids:
                df_joined.append({'case_id': case_id, 'slide_id': ','.join(slide_id)})

        df_temp = pd.DataFrame(df_joined)
        df_temp.to_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\cptac.csv', index=False)



    def test_moving_file(self):

        df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\cptac.csv')

        n_slides = df['slide_id']
        print("slidens number: ", len(n_slides))
        slides = os.listdir(r'E:\Datasets\DP_Pancreas\CPTAC\CPTAC_PDA_v8\PDA')
        dest = r'E:\Datasets\DP_Pancreas\clam_PCTAC'
        os.makedirs(dest, exist_ok=True)

        counter = 0
        for file in n_slides:
            full_file = f"{file}.svs"
            if full_file in slides:
                origin_path = os.path.join('E:\Datasets\DP_Pancreas\CPTAC\CPTAC_PDA_v8\PDA', full_file)
                shutil.move(origin_path, dest)
                counter+=1
        print("Files moved: ", counter)


    def test_prepare4feat_ext(self):
        df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\process_list_autogen.csv')
        df_noext = pd.DataFrame({'slide_id': df['slide_id'].str.split('.').str[0]})
        df_noext.to_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\process_list_autogen_new.csv', index=False)

    def test_check_file(self):

        df_patch = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\process_list_autogen_new.csv')
        df_patch['slide_id'] = df_patch['slide_id'].str.split('.').str[0]
        print(df_patch.shape)

        #files_cartella = os.listdir(r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches_x20_256\patches")
        files_cartella = [os.path.splitext(f)[0] for f in os.listdir(r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\patches")]

        df_filtrato = df_patch[df_patch['slide_id'].isin(files_cartella)]

        df_filtrato.to_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\process_list_autogen_new_filtered.csv', index=False)



    def test_collate_labels(self):

        df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\cptac.csv')

        slides = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\CPTAC_patches\process_list_autogen_new_filtered.csv')

        clin_data = pd.read_csv(r'E:\Datasets\DP_Pancreas\CPTAC\CPTAC_PDA_v8\clinical_data.csv', sep=',')

        clin_df = clin_data[['patient_id', 'df_clinical.vital_status']]
        clin_df = clin_df.rename(columns={'patient_id': 'case_id', 'df_clinical.vital_status': 'label'})
        print(clin_df.shape)
        clin_df = clin_df.loc[clin_df['label'].isin(['Dead', 'Alive'])]
        print(clin_df.shape)
        csv_filtered = pd.merge(df, slides, on='slide_id', how='inner')



        csv_training = pd.merge(csv_filtered, clin_df, on='case_id', how='inner')

        print(csv_training['label'].value_counts())

        csv_training.to_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\dataset_csv\cptac_training.csv', index=False)



    def test_dlbcl(self):
        df = pd.read_csv(r'E:\Datasets\DLBCL\status_casi_CLAM.csv')
        df.dropna(inplace=True)
        df.drop(columns=['EE'], inplace=True)
        #df = df[['Patient_ID', 'COO']]
        df.rename(columns={'Patient_ID': 'sample_id', 'COO': 'label', 'Sample_ID': 'slide_id'}, inplace=True)

        df.to_csv(r'E:\Datasets\DLBCL\df_list.csv', index=False)

        path_svs = r'E:\Datasets\DLBCL\dlbcl_svs'
        slides_id = os.listdir(path_svs)


        data = {'sample_id': [], 'slide_id': [], 'label': []}

        for file in slides_id:
            for index, row in df.iterrows():
                if row['sample_id'] in file:
                    data['sample_id'].append(row['sample_id'])
                    data['slide_id'].append(file)
                    data['label'].append(row['label'])


        df_final = pd.DataFrame(data)
        print("number of slides: ", len(slides_id))
        print("Number of patients: ", df.shape[0])
        print("number of patients + slides: ",df_final.shape[0])
        print(df_final)

        df_final.to_csv(r'E:\Datasets\DLBCL\df_final.csv', index=False)

        print(df_final['label'].value_counts())

        files_non_aggiunti = [file for file in slides_id if file not in df_final['slide_id'].tolist()]
        print("Files svs non aggiunti:")
        print(files_non_aggiunti)


    def test_concat_csv(self):
        genes = ['KRAS', 'TP53', 'RYR1', 'SMAD4', 'TTN', 'ARID1A', 'CDKN2A',
                 'FAT2', 'GLI3', 'MUC16']
        for gene in tqdm(genes):
            folder_path = r"E:\Users\Berloco\PycharmProjects\CLAM\splits\cptac\task_3_HE_vs_mut_100_{}".format(gene)
            for filename in os.listdir(folder_path):
                pattern = r"_\d+\.csv$"
                match = re.search(pattern, filename)
                if match:
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path)
                    newdf = {"train": [],
                    "val": df['val'],
                    "test" : df['test'] }
                    #print(df['train'].shape)
                    concat = pd.concat([df['train'], df['test']], ignore_index=True)
                    newdf['train'] = concat
                    new_df = pd.DataFrame(newdf)
                    new_df = new_df.dropna(how='all')
                    #print(new_df['train'].shape)
                    new_df.to_csv(os.path.join(folder_path, filename), index=False)

                elif filename.endswith("_bool.csv"):
                    df = pd.read_csv(os.path.join(folder_path, filename))
                    df.loc[df['test'], 'train'] = True
                    df.to_csv(os.path.join(folder_path, filename), index=False)

                elif filename.endswith("descriptor.csv"):
                    df = pd.read_csv(os.path.join(folder_path, filename))
                    df['train'] = df['train']+df['test']
                    df.to_csv(os.path.join(folder_path, filename), index=False)


    def test_true_csv(self):
        folder_path = r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\splits\task_3_tumor_mutation_100"

        for filename in os.listdir(folder_path):
            if filename.endswith("_bool.csv"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                df.loc[df['test'], 'train'] = True
                df.to_csv(os.path.join(folder_path, filename), index=False)

    def test_edit_descr(self):
        folder_path = r"E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\splits\task_3_tumor_mutation_100"

        for filename in os.listdir(folder_path):
            if filename.endswith("descriptor.csv"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['train'] = df['train']+df['test']
                df.to_csv(os.path.join(folder_path, filename), index=False)

    def test_prepare_labels(self):
        genes = ['KRAS', 'TP53', 'RYR1', 'SMAD4', 'TTN', 'ARID1A', 'CDKN2A',
       'FAT2', 'GLI3', 'MUC16']

        df = pd.read_csv('./full_datasets/tcga_full_data.csv')
        print(df.columns)
        for gene in genes:
            print(gene)
            df_temp = df[['case_id', 'slide_id', gene]]
            df_temp = df_temp.rename(columns={gene: 'label'})
            df_temp.to_csv('./dataset_csv/tcga_csv/tcga_{}.csv'.format(gene), index=False)


    def test_ensamble(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        # Carica un dataset di esempio
        data = load_iris()
        X, y = data.data, data.target

        # Suddividi il dataset in scripts e test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Addestra un insieme di classificatori
        val = [RandomForestClassifier(n_estimators=10, random_state=i).fit(X_train, y_train) for i in range(5)]

        # Calcola le probabilità predette dai classificatori
        y_score = np.array([c.predict_proba(X_test) for c in val])
        # Calcola la media delle probabilità predette
        y_score_means = np.mean(y_score, axis=0)
        # Predice le etichette basate sulla media delle probabilità
        y_pred = np.argmax(y_score_means, axis=1)

        print("Predizioni:", y_pred)


    def test_prediction(self):
        path = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\multimodal_analysis\multimodal_preds'

        for file in os.listdir(path):
            df_path = os.path.join(path, file)
            df = pd.read_csv(df_path)
            df['Y_hat_multi'] = df.apply(lambda row: 1 if row['mean_p1'] > row['mean_p0'] else 0, axis=1)
            df.to_csv(df_path, index=False)


    def test_prepareDEG(self):

        path_to_label = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\classification_dataset\DEG'
        path_to_data = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\classification_dataset\DEG_NEW'
        dest_dir = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\classification_dataset\DEG_class'
        os.makedirs(dest_dir, exist_ok=True)

        for file in os.listdir(path_to_data):
            file_path = os.path.join(path_to_data, file)

            label_path= os.path.join(path_to_label, file)

            df = pd.read_csv(file_path)
            label = pd.read_csv(label_path)

            #df = df.rename(columns={"Unnamed: 0": "case_id"})
            df = df.T
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.reset_index()
            df = df.rename(columns={"index": "case_id"})

            df['case_id'] = df['case_id'].str.slice(stop=12)
            df = df.merge(label[['case_id', 'label']], on='case_id', how='left')




            df.to_csv(os.path.join(dest_dir, file), index=False)



if __name__ == '__main__':
    unittest.main()
