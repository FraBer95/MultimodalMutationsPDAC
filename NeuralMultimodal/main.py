
import os
import pandas as pd
from NeuralMultimodal.utils import read_multi_sets
from model import training_keras_model
from tqdm import tqdm

from run_eval import inference
from genomic_analysis.voting_classifiers_script.train_voting import training_models
from genomic_analysis.voting_classifiers_script.train_voting_Keras import training_keras_model as keras_naive



if __name__ == '__main__':

    print("Training Neural Multimodal... \n")
    training_path = './dataset/train'
    test_path = './dataset/test'
    logs_path = './logsNaiveNeuralMetricsSMAD4'
    os.makedirs(logs_path, exist_ok=True)
    neural = True

    for gene in tqdm(os.listdir(training_path)):
        if gene == 'SMAD4':
            train_path= os.path.join(training_path, gene)
            test_folder = os.path.join(test_path, gene)

            res_path = os.path.join(logs_path, gene)
            os.makedirs(res_path, exist_ok=True)
            img_df, rna_df = read_multi_sets(train_path, train=True)

            print(f"Training on gene {gene}")


            if neural:
                X_img = img_df.drop(columns=['slide_id'])
                X_rna = rna_df.drop(columns=['case_id', 'label'])
                y = rna_df['label']  # target variable

                classifiers = training_keras_model(X_img, X_rna, y, save_path=res_path, gene=gene)

                img_df_test, rna_df_test = read_multi_sets(test_folder, train=False)

                X_imgTest = img_df_test.drop(columns=['slide_id'])
                X_rnaTest = rna_df_test.drop(columns=['new_case_id', 'label'])
                y = rna_df_test['label'].values  # target variable
                ids = X_rnaTest['case_id']
                inference(X_imgTest, X_rnaTest, y, classifiers, res_path, ids, neural)

            else:

                y = rna_df['label']
                X = pd.concat([img_df, rna_df], axis=1)
                X.drop(columns=['slide_id', 'case_id', 'label'], inplace=True)
                #X.columns = [f"col_{i}" for i in range(X.shape[1])]
                classifiers = training_models(X, y, save_path=res_path)
                #classifiers = keras_naive(X, y, save_path=res_path)

                img_df_test, rna_df_test = read_multi_sets(test_folder, train=False)

                X_imgTest = img_df_test.drop(columns=['slide_id'])
                X_rnaTest = rna_df_test.drop(columns=['new_case_id', 'label'])

                y = rna_df_test['label'].values  # target variable
                X_test = pd.concat([X_imgTest, X_rnaTest], axis=1)
                #X_test.columns = [f"col_{i}" for i in range(X.shape[1])]
                ids = X_rnaTest['case_id']
                #X_test.drop(columns=['slide_id', 'case_id', 'label'], inplace=True)


                inference(X_test, None, y, classifiers, res_path, ids, neural=False)




