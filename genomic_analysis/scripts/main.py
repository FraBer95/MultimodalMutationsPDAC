import pandas as pd
from train import training_models
from plot_curves import save_predictions
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""ML model scripts scripts. 
This script trains three different models: RF, XGB and MLP. 
    1. The scripts data are located in ../classification_dataset/ Classification dataset includes DEG data and AE data,
     both divided by gene mutations.
    2. The test data are located in ../classification_dataset/test_set. full_matrix includes full data of cptac gene expression
     without labels, while full_test_set includes full data of cptac with labels (in order to make feature selection on the same
     genes expression data, according to pre-processing and scripts)

The this script trains model on all csv data files in ../classification_dataset/directory and tests it on the corresponding
data in test set





"""


if __name__ == '__main__':


    AE = False
    if AE:
        print("Training models on autoencoder data... \n")
        training_path = '../datasets/classification_dataset/AE'
        test_path = '../datasets/test_set/AE'
        logs_path = '../logs/AE'
        os.makedirs(logs_path, exist_ok=True)

        for dim_folder in tqdm(os.listdir(training_path)):

            folder = os.path.join(training_path, dim_folder)
            save_path = os.path.join(logs_path, dim_folder)
            test_folder = os.path.join(test_path, dim_folder)
            os.makedirs(save_path, exist_ok=True)

            for filename in tqdm(os.listdir(folder)):

                gene = filename.split('_')[-1].split('.')[0]
                print(f"Training on gene {gene}")
                train_df = pd.read_csv(os.path.join(folder, filename))

                X = train_df.drop(columns=['case_id', 'label'])
                y = train_df['label']  # target variable
                classifiers = training_models(X, y, save_path=save_path)  # scripts models

                test_df = pd.read_csv(os.path.join(test_folder, filename))
                save_predictions(classifiers, test_df, X.columns, 'label', savepath=save_path, AE=True, gene=gene)


    else:
        training_path = '../datasets/classification_dataset/DEG'
        test_path = '../datasets/test_set/DEG'
        logs_path = '../logs_new/DEG'
        os.makedirs(logs_path, exist_ok=True)

        print("Training models on DEG data... \n")

        for filename in  tqdm(os.listdir(training_path)):

            gene = filename.split('_')[-1].split('.')[0]
            save_path = os.path.join(logs_path, gene)

            train_df = pd.read_csv(os.path.join(training_path, filename))

            X = train_df.drop(columns=['case_id', 'label'])
            y = train_df['label'] #target variable
            classifiers = training_models(X, y, save_path=save_path)

            test_df = pd.read_csv(os.path.join(test_path, filename))

            save_predictions(classifiers, test_df, X.columns, 'label', savepath=save_path, AE=False, gene=gene) #validation on test set; if not external label=y_test

