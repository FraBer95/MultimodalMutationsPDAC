import os

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def running_eval_AE(folder, type):

    for gene in os.listdir(folder):
        gene_path = os.path.join(folder, gene)
        metrics = {}
        modelfiles =  os.listdir(gene_path)
        file_dirs = [m for m in modelfiles if m not in 'metrics.csv']
        for modelfile in file_dirs:
            if modelfile not in 'metrics.csv':
                model = modelfile.split('.')[0]

                df_sample = pd.read_csv(os.path.join(gene_path, modelfile))
                df = df_sample.groupby('case_id').agg({
                    'p0': 'mean',
                    'p1': 'mean',
                    'Y': 'first',
                    'Y_hat' : 'mean'
                }).reset_index()


                fpr, tpr, _ = roc_curve(df['Y'], df['p1'])
                roc_auc = roc_auc_score(df['Y'], df['p1'])
                std = df['p1'].std()
                precision, recall, _ = precision_recall_curve(df['Y'], df['p1'])
                average_precision = average_precision_score(df['Y'], df['p1'])

                metrics[f"{model}_{gene}"] = {
                    "roc_auc": roc_auc,
                    "pr_auc": average_precision,
                    "std on p1": std
                }
                metrics_df = pd.DataFrame(metrics).T

            ensambled_path = os.path.join(r'E:\Users\Berloco\PycharmProjects\CLAM\Multimodal_analysis\probs_data', type)
            os.makedirs(ensambled_path, exist_ok=True)



            dims = ['64', '128', '256']
            tokens = gene_path.split(os.sep)
            dim = [t for t in tokens if t in dims]
            filename = f"{model}_{dim[0]}.csv"
            os.makedirs(os.path.join(ensambled_path, tokens[-1]), exist_ok=True)
            df.to_csv(os.path.join(ensambled_path, tokens[-1], filename), index=False)


            metrics_df.to_csv(os.path.join(gene_path, 'metrics.csv'))


def running_eval_DEG(folder, type):

    metrics = {}
    modelfiles = os.listdir(folder)
    file_dirs = [m for m in modelfiles if m not in 'metrics.csv']

    for modelfile in file_dirs:

        if modelfile not in 'metrics.csv':
            model = modelfile.split('.')[0]

            df_sample = pd.read_csv(os.path.join(folder, modelfile))
            df = df_sample.groupby('case_id').agg({
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

            metrics[f"{model}"] = {
                "roc_auc": roc_auc,
                "pr_auc": average_precision,
                "std on p1": std
            }
            metrics_df = pd.DataFrame(metrics).T

        ensambled_path = os.path.join(r'E:\Users\Berloco\PycharmProjects\CLAM\Multimodal_analysis\probs_data', type)
        os.makedirs(ensambled_path, exist_ok=True)


        tokens = folder.split(os.sep)
        filename = f"{model}_{tokens[-1]}.csv"
        os.makedirs(os.path.join(ensambled_path, tokens[-1]), exist_ok=True)
        df.to_csv(os.path.join(ensambled_path, tokens[-1], filename), index=False)

        metrics_df.to_csv(os.path.join(folder, 'metrics.csv'))






if __name__ == '__main__':

    logs_path = '../logs/DEG'
    type = logs_path.split('/')[-1]

    for log_file in os.listdir(logs_path):
        if type == 'AE':
            folder = os.path.join(logs_path, log_file)
            running_eval_AE(folder, type)
        else:
            folder = os.path.join(logs_path, log_file)
            running_eval_DEG(folder, type)