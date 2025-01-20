import os

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
import json





def running_eval_AE(folder, test, features, label, results):

    gene = os.path.split(folder)[-1]
    dim = folder.split(os.sep)[1]
    metrics = {}

    for modelfolder in os.listdir(folder):

        class_dict = {}
        class_list = []

        final_path = os.path.join(results, gene)
        os.makedirs(final_path, exist_ok=True)

        if modelfolder != 'MLPClassifier':
            path_to_model = os.path.join(folder, modelfolder)

            for filename in os.listdir(path_to_model):
                class_list.append(joblib.load(os.path.join(path_to_model, filename)))
            class_dict[modelfolder] = class_list

        else:
            path_to_model = os.path.join(folder, modelfolder)
            for filename in os.listdir(path_to_model):
                class_list.append(load_model(os.path.join(path_to_model, filename)))
            class_dict[modelfolder] = class_list

        X_test = test[features[1:len(features)]].values
        y_test = test[[label]].values

        for key, val in class_dict.items():

            if modelfolder == 'MLPClassifier':
                modelfolder = 'MLP'
                y_score = np.array([c.predict(X_test) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.argmax(y_score_means, axis=1)

                p0 = 1 - y_score_means
                new_df_means = np.hstack((p0, y_score_means))
                y_pred_df = pd.DataFrame(data=new_df_means, index=test['case_id'].index, columns=['p0', 'p1'])

            else:
                if modelfolder == 'RandomForestClassifier': modelfolder = 'RF'
                else: modelfolder = 'XGBClassifier'
                y_score = np.array([c.predict_proba(X_test) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.argmax(y_score_means, axis=1)
                y_pred_df = pd.DataFrame(data=y_score_means, index=test['case_id'].index, columns=['p0', 'p1'])

            predict_df = pd.concat([test['case_id'], y_pred_df, pd.DataFrame(y_test[:, 0], columns=['Y']),
                                    pd.DataFrame(y_pred, columns=['Y_hat'])], axis=1)

            #predict_df.to_csv(os.path.join(final_path, f'{modelfolder}_{dim}.csv'), index=False)

            df = predict_df.groupby('case_id').agg({
                'p0': 'mean',
                'p1': 'mean',
                'Y': 'first',
                'Y_hat': 'mean'
            }).reset_index()


            df.to_csv(os.path.join(final_path, f'{modelfolder}_{dim}.csv'), index=False)

            fpr, tpr, _ = roc_curve(df['Y'], df['p1'])
            roc_auc = roc_auc_score(df['Y'], df['p1'])
            std = df['p1'].std()
            precision, recall, _ = precision_recall_curve(df['Y'], df['p1'])
            average_precision = average_precision_score(df['Y'], df['p1'])

            metrics[f"{modelfolder}"] = {
                "roc_auc": roc_auc,
                "pr_auc": average_precision,
                "std on p1": std
            }
    metrics_df = pd.DataFrame(metrics).T
    #metrics_df.to_csv(os.path.join(final_path, 'subject_metrics.csv'))




def running_eval_DEG(folder, test, features, label, results): #folder = DEG+Gene -> MODELLO


    gene = os.path.split(folder)[-1]
    metrics = {}

    for modelfolder in os.listdir(folder):

        class_dict = {}
        class_list = []

        final_path = os.path.join(results, gene)
        os.makedirs(final_path, exist_ok=True)

        if modelfolder != 'MLPClassifier':
            path_to_model = os.path.join(folder, modelfolder)

            for filename in os.listdir(path_to_model):
                class_list.append(joblib.load(os.path.join(path_to_model, filename)))
            class_dict[modelfolder] = class_list

        else:
            path_to_model = os.path.join(folder, modelfolder)
            for filename in os.listdir(path_to_model):
                class_list.append(load_model(os.path.join(path_to_model, filename)))
            class_dict[modelfolder] = class_list


        X_test = test[features[1:len(features)]].values
        y_test = test[[label]].values

        X_log = np.log1p(X_test)
        X_scaled = StandardScaler().fit_transform(X_log)

        for key, val in class_dict.items():


            if modelfolder == 'MLPClassifier':
                modelfolder = 'MLP'
                y_score = np.array([c.predict(X_scaled) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.where(y_score_means > 0.5, 1, 0)

                p0 = 1-y_score_means
                new_df_means = np.hstack((p0, y_score_means))

                y_pred_df = pd.DataFrame(data=new_df_means, index=test['case_id'].index, columns=['p0', 'p1'])

            else:
                if modelfolder == 'RandomForestClassifier': modelfolder = 'RF'
                else: modelfolder = 'XGBClassifier'
                y_score = np.array([c.predict_proba(X_scaled) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.argmax(y_score_means, axis=1)
                y_pred_df = pd.DataFrame(data=y_score_means, index=test['case_id'].index, columns=['p0', 'p1'])


            predict_df = pd.concat([test['case_id'], y_pred_df, pd.DataFrame(y_test[:, 0], columns=['Y']),
                                    pd.DataFrame(y_pred, columns=['Y_hat'])], axis=1)


            predict_df.to_csv(os.path.join(final_path, f'{modelfolder}.csv'), index=False)

            df = predict_df.groupby('case_id').agg({
                'p0': 'mean',
                'p1': 'mean',
                'Y': 'first',
                'Y_hat': 'mean'
            }).reset_index()

            fpr, tpr, _ = roc_curve(df['Y'], df['p1'])
            roc_auc = roc_auc_score(df['Y'], df['p1'])
            std = df['p1'].std()
            precision, recall, _ = precision_recall_curve(df['Y'], df['p1'])
            average_precision = average_precision_score(df['Y'], df['p1'])

            metrics[f"{modelfolder}"] = {
                "roc_auc": roc_auc,
                "pr_auc": average_precision,
                "std on p1": std
            }
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(final_path, 'subject_metrics.csv'))







if __name__ == '__main__':

    AE = False

    results_folder = r'./evaluation_results/'

    if AE:
        type = 'AE'
        logs_path = './log_voting/AE'
    else:
        type  = 'DEG'
        logs_path = './log_voting/DEG'


    for log_file in os.listdir(logs_path):
        if AE:
            dim = log_file
            path_to_dim = os.path.join(logs_path, log_file)
            results = os.path.join(results_folder, type)
            os.makedirs(results, exist_ok=True)


            for gene in os.listdir(path_to_dim):
                test = pd.read_csv( r'E:\Users\\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\AE\{}\AE_{}.csv'.format(
                    dim, gene))
                folder = os.path.join(path_to_dim, gene)
                running_eval_AE(folder, test, features=test.columns[0:len(test.columns) - 1], label='label',
                             results=results)

        else:                                           #DEG/GENE/MODELLO
            gene = log_file

            df = pd.read_csv(r'E:\Users\\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\test_set\DEG\DEG_{}.csv'.format(gene))
            df_train = pd.read_csv( r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\classification_dataset\DEG\DEG_{}.csv'.format(gene))


            features_list = df_train.columns[0:len(df_train.columns)]

            test = df[features_list]

            folder = os.path.join(logs_path, log_file)

            results = os.path.join(results_folder, type)
            os.makedirs(os.path.join(results, type), exist_ok=True)

            running_eval_DEG(folder, test, features=test.columns[0:len(test.columns)-1], label='label', results=results)