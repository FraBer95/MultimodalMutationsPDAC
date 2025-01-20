import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, \
    accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import seaborn as sns
import os
import numpy as np
import json

def conf_matrix():
    df = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\DPMultimodal\CLAM\eval_results\EVAL_tcga_eval_tcga_x20_fce\fold_7.csv')
    conf_matrix = confusion_matrix(df['Y'], df['Y_hat'])
    accuracy = accuracy_score(df['Y'], df['Y_hat'])
    print("Accuracy:", accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def average_metrics(folder_path, genes, dest_dir):
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    fold = folder_path.split('_')[-2:]
    metrics_dict = {}
    if fold[0] in genes:

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv") and filename not in ['summary.csv']:
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['case_id']=df['slide_id'].str[:-3]
                df.drop('slide_id', axis=1, inplace=True)
                df = df.groupby('case_id').agg({
                    'p_0': 'mean',
                    'p_1': 'mean',
                    'Y': 'first',
                    'Y_hat': 'mean'
                }).reset_index()
                df['Y_hat'] = df['Y_hat'].round()

                precision = precision_score(df['Y'], df['Y_hat'], zero_division=0)
                recall = recall_score(df['Y'], df['Y_hat'], zero_division=0)
                f1 = f1_score(df['Y'], df['Y_hat'], zero_division=0)

                roc_auc = roc_auc_score(df['Y'], df['p_1'])

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                roc_auc_list.append(roc_auc)

        mean_precision = sum(precision_list) / len(precision_list)
        mean_recall = sum(recall_list) / len(recall_list)
        mean_f1 = sum(f1_list) / len(f1_list)
        mean_roc_auc = sum(roc_auc_list) / len(roc_auc_list)

        print(f'*****Metrics for fold: {fold}*****')
        print("Mean Precision:", mean_precision)
        print("Mean Recall:", mean_recall)
        print("Mean F1-score:", mean_f1)
        print("Mean ROC AUC:", mean_roc_auc)

        metrics_dict[f"{fold[0]}_{fold[1]}"] = {
                        "precision" : mean_precision,
                        "recall" : mean_recall,
                        "mean_f1" : mean_f1,
                        "mean_roc_auc" : mean_roc_auc}

        metrics_file = os.path.join(dest_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as json_file:
                existing_metrics = json.load(json_file)
        else:
            existing_metrics = {}

            # Update the existing metrics with new metrics
        existing_metrics.update(metrics_dict)

        # Write the updated metrics back to the file
        with open(metrics_file, "w") as json_file:
            json.dump(existing_metrics, json_file, indent=4)


def plot_roc_curves(folder_path, genes, dest_dir):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold = folder_path.split('_')[-2:]
    if fold[0] in genes:

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv") and filename not in ['summary.csv']:
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['case_id']=df['slide_id'].str[:-3]
                df.drop('slide_id', axis=1, inplace=True)
                df = df.groupby('case_id').agg({
                    'p_0': 'mean',
                    'p_1': 'mean',
                    'Y': 'first',
                    'Y_hat': 'mean'
                }).reset_index()
                df['Y_hat'] = df['Y_hat'].round()
                fpr, tpr, _ = roc_curve(df['Y'], df['p_1'])
                roc_auc = roc_auc_score(df['Y'], df['p_1'])

                plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (len(tprs) + 1, roc_auc))

                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

                aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {fold[0]}_{fold[1]}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(dest_dir, 'ROC_curves_{}_{}.png'.format(fold[0], fold[1])))
        plt.close()



def single_metrics(folder_path, genes, dest_dir):

    fold = os.path.split(folder_path)[-1].split('_')
    model, gene = f"{fold[1]}_{fold[-2]}", fold[-1].split('.')[0]
    if folder_path.endswith(".csv"):
        df = pd.read_csv(folder_path)
        metrics_dict = {}
        precision = precision_score(df['Y'], df['Y_hat_multi'], zero_division=0)
        recall = recall_score(df['Y'], df['Y_hat_multi'], zero_division=0)
        f1 = f1_score(df['Y'], df['Y_hat_multi'], zero_division=0)

        roc_auc = roc_auc_score(df['Y'], df['mean_p1'])



        print(f'*****Metrics for fold: {fold}*****')
        print(" Precision:", precision)
        print(" Recall:", recall)
        print(" F1-score:", f1)
        print(" ROC AUC:", roc_auc)

        metrics_dict[f"{model}_{gene}"] = {
                        "precision" : precision,
                        "recall" : recall,
                        "mean_f1" : f1,
                        "mean_roc_auc" : roc_auc}

        metrics_file = os.path.join(dest_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as json_file:
                existing_metrics = json.load(json_file)
        else:
            existing_metrics = {}


        existing_metrics.update(metrics_dict)


        with open(metrics_file, "w") as json_file:
            json.dump(existing_metrics, json_file, indent=4)


def plot_roc_curves_single(file_folder, dest_dir):

    metrics = {}
    files = os.listdir(file_folder)
    files = [f for f in files if not f.startswith('.')]
    for filename in files:
        file_path = os.path.join(file_folder, filename )

        #fold = os.path.split(folder_path)[-1].split('_')
        #model, gene = f"{fold[1]}_{fold[2]}", fold[-1].split('.')[0]
        model_gene = os.path.split(file_path)[-1].split('_')[0]

        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError as E:
                print("Exception: ", E)
                print(file_path)

            if 'mean_p1' not in df.columns: #no multimodale
                if 'p1' in df.columns: #modello transcrittomico
                    fpr, tpr, _ = roc_curve(df['Y'], df['p1'])
                    roc_auc = roc_auc_score(df['Y'], df['p1'])
                    std = df['p1'].std()
                elif 'p_1'in df.columns: #modello imaging
                    fpr, tpr, _ = roc_curve(df['Y'], df['p_1'])
                    roc_auc = roc_auc_score(df['Y'], df['p_1'])
                    std = df['p_1'].std()
            else:
                fpr, tpr, _ = roc_curve(df['Y'], df['mean_p1'])
                roc_auc = roc_auc_score(df['Y'], df['mean_p1'])
                std = df['mean_p1'].std()


            # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Curve (AUC = %0.2f)' % roc_auc)
            #
            # plt.plot([0, 1], [0, 1], color='b', linestyle='--', label='Random Guessing')
            #
            #
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(f'ROC Curve for {model_gene}')
            # plt.legend(loc="lower right")

            #plt.savefig(os.path.join(dest_dir, 'ROC_curve_{}.png'.format(model_gene)))
            #plt.close()

            if 'mean_p1' not in df.columns: #no multimodale
                if 'p1' in df.columns: #modello transcrittomico
                    precision, recall, _ = precision_recall_curve(df['Y'], df['p1'])
                    average_precision = average_precision_score(df['Y'], df['p1'])
                elif 'p_1'in df.columns: #modello imaging
                    precision, recall, _ = precision_recall_curve(df['Y'], df['p_1'])
                    average_precision = average_precision_score(df['Y'], df['p_1'])
            else:
                precision, recall, _ = precision_recall_curve(df['Y'], df['mean_p1'])
                average_precision = average_precision_score(df['Y'], df['mean_p1'])


            # Plot della Precision-Recall curve
            # plt.figure()
            # plt.plot(recall, precision, lw=1, alpha=0.3, label='PR Curve (AP = %0.2f)' % average_precision)
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title(f'Precision-Recall Curve for {model_gene}')
            # plt.legend(loc="lower left")
            #plt.savefig(os.path.join(dest_dir, 'PR_curve_{}.png'.format(model_gene)))
            #plt.close()

            metrics[model_gene] = {
                    "roc_auc": roc_auc,
                    "pr_auc": average_precision,
                    "std on p1": std
            }

    return metrics










if __name__ == '__main__':
    main_fold = r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\logs\VAE\ensamble_predictions'
    #genes = ['KRAS', 'TP53', 'SMAD4', 'TTN', 'CDKN2A']
    log_fold = './RESULTS_CURVES_AE'
    dest_dir = os.path.join(log_fold, 'KRASAE_Genomic_metrics_subject_level')
    os.makedirs(dest_dir, exist_ok=True)
    #for folder in os.listdir(main_fold):
    #    folder = os.path.join(main_fold, folder)
        #conf_matrix()
        #single_metrics(folder, genes, dest_dir)
    metrics = plot_roc_curves_single(main_fold, dest_dir)