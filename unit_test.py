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
        genes = ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']
        for gene in tqdm(genes):
            folder_path = r"E:\Users\Berloco\PycharmProjects\CLAM\splits\tcga\task_3_wt_vs_mut_100_{}".format(gene)
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

        df = pd.read_csv('full_datasets/tcga_full_data.csv')
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


    def test_prepareWSI(self):
        import torch
        from collections import defaultdict

        # Percorso della cartella contenente i file .pt
        input_dir = "/mnt/d/Datasets/Pancreas_Features/CLAM_TCGA_features_x20_CONCH/pt_files"
        output_dir = "/mnt/d/Datasets/Pancreas_Features/CLAM_TCGA_features_x20_CONCH/pt_files_new"
        os.makedirs(output_dir, exist_ok=True)

        case_tensors = defaultdict(list)


        def extract_case_id(filename):
            return "-".join(filename.split("-")[:3])

        file_groups = defaultdict(list)
        for file in os.listdir(input_dir):
            if file.endswith(".pt"):  # Considera solo i file .pt
                case_id = extract_case_id(file)
                file_groups[case_id].append(file)

        # Filtra solo i case_id con duplicati
        duplicate_case_ids = {case_id: files for case_id, files in file_groups.items() if len(files) > 1}

        # Passo 2: Concatenare e salvare solo i tensori duplicati
        for case_id, files in duplicate_case_ids.items():
            tensors = []
            for file in files:
                file_path = os.path.join(input_dir, file)
                tensors.append(torch.load(file_path))  # Carica solo i tensori necessari

            # Concatena i tensori lungo la dimensione 0
            concatenated_tensor = torch.cat(tensors, dim=0)

            output_path = os.path.join(output_dir, f"{case_id}_concatenated.pt")
            torch.save(concatenated_tensor, output_path)
            print(f"Salvato il tensore concatenato per {case_id} in {output_path}")


    def test_renameWSIConcat(self):
        origin_path = "/mnt/e/Users/Berloco/PycharmProjects/CLAM/dataset_csv/tcga_csv"
        csv_name = 'tcga_TP53.csv'
        csv_path = os.path.join(origin_path, csv_name)
        concatenated_tensors_dir = "/mnt/d/Datasets/Pancreas_Features/CLAM_TCGA_features_x20_CONCH/pt_files_new"

        # Carica il file CSV
        df = pd.read_csv(csv_path)

        # Trova i duplicati basati sul case_id
        duplicated_case_ids = df[df.duplicated(subset=['case_id'], keep=False)]['case_id'].unique()

        # Itera sui case_id duplicati
        for case_id in duplicated_case_ids:
            # Filtra le righe associate al case_id corrente
            case_rows = df[df['case_id'] == case_id]

            # Percorso del tensore concatenato
            concatenated_tensor_path = os.path.join(concatenated_tensors_dir, f"{case_id}_concatenated.pt")

            if os.path.exists(concatenated_tensor_path):


                # Aggiorna lo slide_id con il nuovo identificativo concatenato
                concatenated_slide_id = f"{case_id}_concatenated"
                df.loc[df['case_id'] == case_id, 'slide_id'] = concatenated_slide_id
            else:
                print(f"Tensore concatenato non trovato per {case_id}. Mantenendo i valori originali.")

        # Rimuovi i duplicati mantenendo una sola riga per case_id
        df = df.drop_duplicates(subset=['case_id'], keep='first')

        # Salva il nuovo file CSV
        df.to_csv(csv_path, index=False)

    def test_filtering_csvWSI(self):
         origin_path = "/mnt/e/Users/Berloco/PycharmProjects/CLAM/dataset_csv/tcga_csv"
         csv_name = 'tcga_SMAD4.csv'
         original_csv_path = os.path.join(origin_path, csv_name)
         transcriptomic_csv_path = "/mnt/e/Users/Berloco/PycharmProjects/CLAM/Trascriptomics_analysis/datasets/autoencoder_data/raw_data/TCGA_5kgenes.csv"  # CSV con sample_id trascrittomici
         #filtered_csv_path = "/path/to/final_filtered.csv"  # CSV filtrato

         # Carica i CSV
         original_df = pd.read_csv(original_csv_path)
         tcga5k = pd.read_csv(transcriptomic_csv_path)

         transcriptomic_df = pd.DataFrame()
         # Converti i sample_id trascrittomici sostituendo '.' con '-'
         transcriptomic_df['sample_id'] = tcga5k.iloc[:, 0]

         # Filtra il CSV originale mantenendo solo i case_id che sono prefissi dei sample_id trascrittomici
         valid_case_ids = transcriptomic_df['sample_id'].str.split('.').str[:3].str.join('-').unique()

         filtered_df = original_df[original_df['case_id'].isin(valid_case_ids)]

         # Salva il risultato
         filtered_df.to_csv(original_csv_path, index=False)

         print(f"Filtraggio completato")


    def test_prepareDataTest(self):
        import pandas as pd

        encoded_test_path = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\autoencoder_data\encoded_data\CPTAC' #192, full data
        out_dir = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\autoencoder_data\encoded_data\CPTAC_new1'
        os.makedirs(out_dir, exist_ok=True)
        df_full = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\full_matrix\rna_cptac.csv') #192

        df_pairs = pd.read_json(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\TCIA CPTAC Pathology Portal.json')
        df_pairs = df_pairs[df_pairs.Specimen_Type == 'tumor_tissue'] #382

        ids = pd.DataFrame(df_full['ID'])  # 192 con concatenazione
        df_full.rename(columns={'ID': 'case_id'}, inplace=True)

        files = os.listdir(encoded_test_path)
        dict_encoded = {}
        for file in files:

            df_encoded = pd.read_csv(os.path.join(encoded_test_path, file)) #192
            df_encoded['new_case_id'] = ids #specimen_Id

            df_encoded['new_case_id'] = df_encoded['new_case_id'].str.split(';')
            df_exploded = df_encoded.explode('new_case_id', ignore_index=True)

            enc_filtered = df_exploded[df_exploded['new_case_id'].isin(df_pairs['Specimen_ID'])]
            #enc_filtered.to_csv(os.path.join(out_dir, file), index=False) #146
            dict_encoded[file] = enc_filtered

        filtered_df_pairs = df_pairs[df_pairs['Specimen_ID'].isin(enc_filtered['new_case_id'])]
        for gene in ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']:

            df_wsi = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\dataset_csv\cptac_csv\cptac_{}.csv'.format(gene))
            wsi_new = df_wsi[df_wsi['slide_id'].isin(filtered_df_pairs['Slide_ID'])] #filtro quelli con specimen associato

            wsi_new.to_csv(os.path.join(out_dir, 'cptac_{}.csv'.format(gene)), index=False)

        WSIfiltered_df_pairs = filtered_df_pairs[filtered_df_pairs['Slide_ID'].isin(df_wsi['slide_id'])]
        #clean-up per rna seq
        for name, item in dict_encoded.items():
            enc_filtered = item[item['new_case_id'].isin(WSIfiltered_df_pairs['Specimen_ID'])]
            enc_filtered.to_csv(os.path.join(out_dir, name), index=False)  #146




    def test_prepareRNATestAE(self):
        import pandas as pd

        dim_list = ['64', '128', '256']
        gene_list = ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']
        path_to_clinical = r'G:\Berloco\DP_Pancreas\CPTAC\CPTAC_PDA_v8\clinical_data.csv'
        path_to_test = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\autoencoder_data\encoded_data\CPTAC_new1'
        df = pd.read_csv(path_to_clinical)
        dest_dir = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\NEW_AE'

        for dim in dim_list:
            test_df_filtered = pd.read_csv(os.path.join(path_to_test, 'CPTAC_AE_5kgenes_{}.csv'.format(dim)))
            for gene in gene_list:
                labels = df[['patient_id', gene]]
                labels.drop_duplicates(inplace=True)
                labels = labels.rename(columns={'patient_id': 'case_id', gene: 'label'})
                new_ae = test_df_filtered.merge(labels, on='case_id', how='inner')
                #new_ae.dropna()
                dest_path = os.path.join(dest_dir, dim)
                os.makedirs(dest_path, exist_ok=True)
                new_ae.to_csv(os.path.join(dest_path, 'AE_{}.csv'.format(gene)), index=False)



    def test_prepareRNATestDEG(self):
            import pandas as pd
            gene_list = ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']
            path_to_clinical = r'G:\Berloco\DP_Pancreas\CPTAC\CPTAC_PDA_v8\clinical_data.csv'
            full_cptac = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\full_matrix\rna_cptac.csv')
            full_cptac.rename(columns={'ID': 'case_id'}, inplace=True)

            encoded = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\NEW_AE\64\AE_KRAS.csv')
            ids = encoded['new_case_id']

            full_cptac['new_case_id'] = full_cptac['case_id'].str.split(';')
            df_exploded = full_cptac.explode('new_case_id', ignore_index=True)
            #df_exploded.drop(columns=['case_id'], inplace=True)
            #df_exploded.rename(columns={'new_case_id': 'case_id'}, inplace=True)
            filtered = df_exploded[df_exploded['new_case_id'].isin(ids)]
            filtered['case_id'] = filtered['case_id'].str[:9]

            df = pd.read_csv(path_to_clinical)

            for gene in gene_list:
                labels = df[['patient_id', gene]]
                labels.drop_duplicates(inplace=True)
                labels = labels.rename(columns={'patient_id': 'case_id', gene: 'label'})
                new_deg = filtered.merge(labels, on='case_id', how='inner')

                #new_deg.drop(columns=['case_id'], inplace=True)
                #new_deg.rename(columns={'new_case_id': 'case_id'}, inplace=True)
                new_deg.to_csv(os.path.join(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\NEW_DEG', 'DEG_{}.csv'.format(gene)), index=False)


    def test_metrics(self):

        import os
        dims = [64, 128, 256]
        for dim in dims:
            gen_folder = os.path.join(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\voting_classifiers_script\log_votingTCGA\AE', str(dim))
            for dir in os.listdir(gen_folder):
                path_to_folder = os.path.join(gen_folder, dir)
                dict_models = {"Model": [], "AUROC": [], "AUPRC": []}
                for file in os.listdir(path_to_folder):
                    if file not in ('Allmetrics.csv'):
                        df = pd.read_csv(os.path.join(path_to_folder, file))
                        df.drop(columns=['Unnamed: 0'], inplace=True)
                        dict_models["Model"].extend(df["Model"].round(5).tolist())
                        dict_models["AUROC"].extend(df["AUROC"].round(5).tolist())
                        dict_models["AUPRC"].extend(df["AUPRC"].round(5).tolist())
                df_metrics = pd.DataFrame(dict_models).T
                df_metrics.to_csv(os.path.join(path_to_folder, 'Allmetrics.csv'))


if __name__ == '__main__':
    unittest.main()
