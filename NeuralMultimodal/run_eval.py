import os

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np



def inference(testImg, testRna, y, class_dict, res_path, ids, neural):

        if neural:
            testRna.drop(columns=['case_id'], inplace=True)
            #testImg.drop(columns=['case_id'], inplace=True)
        else:
            testImg.drop(columns=['case_id'], inplace=True)

        metrics = {}

        for key, val in class_dict.items():

            modelfolder = key
            if neural:
                y_score = np.array([c.predict([testImg.values, testRna.values]) for c in val])
                #y_score = np.array([c.predict([testImg.values]) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.where(y_score_means > 0.5, 1, 0)
                p0 = 1 - y_score_means
                new_df_means = np.hstack((p0, y_score_means))
                y_pred_df = pd.DataFrame(data=new_df_means, index=ids.index, columns=['p0', 'p1'])
            else:
                y_score = np.array([c.predict_proba(testImg.values) for c in val])
                y_score_means = np.mean(y_score, axis=0)
                y_pred = np.argmax(y_score_means, axis=1)
                y_pred_df = pd.DataFrame(data=y_score_means, index=ids.index, columns=['p0', 'p1'])



            predict_df = pd.concat([ids, y_pred_df, pd.DataFrame(y, columns=['Y']),
                                    pd.DataFrame(y_pred, columns=['Y_hat'])], axis=1)


            predict_df.to_csv(os.path.join(res_path, f'{modelfolder}.csv'), index=False)

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
            metrics_df = pd.DataFrame(metrics).round(decimals=5)
            metrics_df.to_csv(os.path.join(res_path, 'Testsubject_metrics{}.csv'.format(modelfolder)))
