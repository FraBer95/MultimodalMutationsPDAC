import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from plot_curves import save_predictions
import shap
import joblib
from genomic_analysis.AE.utils import *
from genomic_analysis.AE.autoencoder import build_autoencoder
import matplotlib.pyplot as plt


def get_classifiers(full_path):

    class_dict = {}
    for classifier in os.listdir(full_path):
        class_list = []
        classifiers_path = os.path.join(full_path, classifier)
        models = os.listdir(classifiers_path)
        for model in models:

            class_list.append(joblib.load(os.path.join(classifiers_path, model)))

        class_dict[classifier] = class_list

    return class_dict


def predict(encoder, classifier_dict, data):
    encoded_data = encoder.predict(data)

    y_score = np.array([c.predict_proba(encoded_data) for c in classifier_dict])
    y_score_means = np.mean(y_score, axis=0)
    y_pred = np.argmax(y_score_means, axis=1)
    return y_pred


if __name__ == "__main__":

    path_to_models = r'./log_voting/AE'
    test_df = pd.read_csv('../datasets/autoencoder_data/raw_data/cptac_5kgenes.csv')
    train_df = pd.read_csv('../datasets/autoencoder_data/raw_data/TCGA_5kgenes.csv')
    result_path = r'./results'
    os.makedirs(result_path, exist_ok=True)


    for dim_dir in os.listdir(path_to_models):
        path_dir_genes = os.path.join(path_to_models, dim_dir)
        for gene_dir in os.listdir(path_dir_genes):
            full_path = os.path.join(path_dir_genes, gene_dir)
            classifiers_dict = get_classifiers(full_path) #load classifiers


            single_result_path = os.path.join(result_path, dim_dir, gene_dir)
            os.makedirs(single_result_path, exist_ok=True)

            #pre-processing

            tr_data, tr_sample_ids = prepare_data(train_df)
            data_norm_tr = normalize_data(tr_data)

            data, sample_ids = prepare_data(test_df)
            data_norm = normalize_data(data)

            data_norm_df = pd.DataFrame(data=data_norm, columns=data.columns)


            ckpt_dir = os.path.join('../AE/models', str(dim_dir))
            ckpt_path = os.path.join(ckpt_dir, 'cp.ckpt')
            encoder, autoencoder = build_autoencoder(data.shape[1], hidden_dim=int(dim_dir))

            autoencoder.load_weights(ckpt_path).expect_partial()

            for layer in encoder.layers:
                layer.trainable = False

        for key, values in classifiers_dict.items():
            explainer = shap.KernelExplainer(
                lambda x: predict(encoder, classifiers_dict[key], x), data_norm_tr
            )

            shap_values = explainer.shap_values(data_norm_df.iloc[1, :])

            shap.force_plot(explainer.expected_value, shap_values, data_norm_df.iloc[1, :], feature_names=data_norm_df.columns, matplotlib=True)

                #plt.savefig(os.path.join(result_path, f'{key}_shap_force_plot.png'), format='png', dpi=300, bbox_inches='tight')
            #plt.close()



            pass
"""

            sciAE = KerasClassifier(model=encoder)

            for classifier in classifiers_list:

                pipeline = Pipeline([
                    ('encoder', sciAE),
                    ('classifier', classifier)
                                    ])

                #y_scores = save_predictions(pipeline, data_norm, savepath=single_result_path, AE=False,
                 #                gene=dir)  # validation on test set; if not external label=y_test

                explainer = shap.KernelExplainer(predict(encoder, classifier, data_norm), )


"""
