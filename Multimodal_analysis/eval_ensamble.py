import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

def eval_metrics(path, dest):
    metrics = {}
    for dir in os.listdir(path):
        path_to_dir = os.path.join(path, dir)
        for filename in os.listdir(path_to_dir):

            df = pd.read_csv(os.path.join(path_to_dir, filename))
            model = filename.split('.')[0]

            precision, recall, _ = precision_recall_curve(df['Y'], df['mean_p1'])
            average_precision = average_precision_score(df['Y'], df['mean_p1'])
            roc_auc = roc_auc_score(df['Y'], df['mean_p1'])
            std = df['mean_p1'].std()

            metrics[f"{model}"] = {
                "roc_auc": roc_auc,
                "pr_auc": average_precision,
                "std on mean_p1": std
            }

        metrics_df = pd.DataFrame(metrics).T
        metrics_path = os.path.join(dest, dir)

        metrics_df = metrics_df.sort_values(by='roc_auc', ascending=False)

        os.makedirs(metrics_path, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_path, "metrics.csv"))


        # Plotting
        ax = metrics_df.plot(kind='barh', figsize=(12, 8), legend=False)

        plt.title('Metrics per Model-Gene Pair')
        plt.xlabel('Score')
        plt.ylabel('Model-Gene Pair')
        plt.xticks(rotation=45, ha='right')
        plt.xlim(0, 1)


        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=10)  # Ridimensioniamo la dimensione del testo

        # Creiamo una legenda personalizzata in basso a sinistra
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(metrics_path,'metrics.png'))
        plt.close()
        plt.show()







if __name__ == '__main__':

    path = './multimodal_preds_new/'
    dest_path = './multimodal_metrics_new/'
    os.makedirs(dest_path, exist_ok=True)
    eval_metrics(path, dest_path)