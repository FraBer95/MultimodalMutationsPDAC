import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

from genomic_analysis.AE.utils import *
from genomic_analysis.AE.autoencoder import build_autoencoder
from XAI_utils import *
import pickle


if __name__ == "__main__":

    Flag_autoencoder = False
    load_explain = True
    local_XAI = True


    if not Flag_autoencoder:
         gene_model = {
             'CDKN2A': "MLPClassifier",
              'KRAS': "RandomForestClassifier",
              'TP53': "MLPClassifier",
              'SMAD4': "XGBClassifier"
                       }


    else:
        gene_model = {
            'CDKN2A': ["MLPClassifier", 256],
             'KRAS': ["MLPClassifier", 256],
             'TP53': ["MLPClassifier", 64],
             'SMAD4': ["MLPClassifier", 256]
                      }

    if local_XAI:
        samples_data = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\heatmaps\process_lists\heatmap_CDKN2A_dataset.csv')
        sample_list = samples_data['slide_id'].str.slice(stop=9).to_list()
    else:
        sample_id = ''


    for gene, model_name in gene_model.items():

            print(f"Creating explanation for: {gene} with {model_name}")

            if Flag_autoencoder:
                print("Loading data and Autoencoders...")
                path_to_ML_model = f'./log_voting/AE/{model_name[1]}/{gene}/{model_name[0]}'
                dim = model_name[1]
                df = pd.read_csv('../datasets/autoencoder_data/raw_data/cptac_5kgenes.csv')
                result_path = f'./results/AE'

            else:
                print("Loading DEG data and ML Model...")
                dim = ""
                path_to_ML_model = f'./log_voting/DEG/{gene}/{model_name}'
                df = pd.read_csv(f'../datasets/test_set/DEG/DEG_{gene}.csv')
                df_train = pd.read_csv(f'../datasets/classification_dataset/DEG/DEG_{gene}.csv')
                features_list = df_train.columns[0:len(df_train.columns) - 1]
                result_path = f'./results/DEG_new'



            conversion_table = pd.read_csv(r'E:\Users\Berloco\PycharmProjects\CLAM\genomic_analysis\datasets\converted_genes_DEG.csv')

            os.makedirs(result_path, exist_ok=True)
            classifiers_dict = get_classifiers(path_to_ML_model) #load classifiers

            single_result_path = os.path.join(result_path, gene, str(dim))


            os.makedirs(single_result_path, exist_ok=True)


            if Flag_autoencoder:

                data, sample_ids = prepare_data(df)
                data_norm = normalize_data(data)


                data_norm_df = pd.DataFrame(data=data_norm, columns=data.columns)
                print("\n Mapping genes...")
                test_df = rename_genes(conversion_table, data_norm_df, df.columns)
                if local_XAI: XAI_sample_ID  = get_sample(id_pat=sample_id, id_list=sample_ids, df=data_norm_df)

            else:

                df = df[features_list]
                print("\n Mapping genes...")
                test_df = rename_genes(conversion_table, df, df.columns[1:])
                #if local_XAI:
                   # XAI_sample_ID, sample_index = get_sample(id_pat=sample_id, df=test_df)



        #AE processing

            if Flag_autoencoder:
                print("Loading autoencoder...\n")

                ckpt_dir = os.path.join('../AE/models', str(dim))
                ckpt_path = os.path.join(ckpt_dir, 'cp.ckpt')
                encoder, autoencoder = build_autoencoder(data.shape[1], hidden_dim=int(dim))

                autoencoder.load_weights(ckpt_path).expect_partial()
                for layer in encoder.layers:
                    layer.trainable = False

                if model_name[0] == 'MLPClassifier':
                    classifiers_list = create_models(encoder, classifiers_dict)


                #create explanation

                key = os.path.split(path_to_ML_model)[-1]
                if not (load_explain):
                    print("Computing Shapley values", '\n')
                    #test = test_df.drop(columns=['case_id'])
                    explanation_list = []

                    #test = test_df.drop(columns=['case_id'])

                    if model_name[0] == 'MLPClassifier':
                        for model in classifiers_list:
                            explainer = shap.DeepExplainer(model, test_df.to_numpy())
                            explanation_list.append(explainer(test_df.to_numpy()))

                        with open(os.path.join(single_result_path, f'{key}_ExplanationList_deep.pkl'), 'wb') as file:
                            pickle.dump(explanation_list, file)


                    else:
                        for key, values in classifiers_dict.items():
                            explainer = shap.KernelExplainer(
                                lambda x: predict(encoder, classifiers_dict[key], x), test_df, link="logit"
                            )
                            explanation = explainer(test_df)

                        with open(os.path.join(single_result_path, f'{key}_Explanation.pkl'), 'wb') as file:
                            pickle.dump(explanation_list, file)


                if load_explain:
                    with open(os.path.join(single_result_path, f'{key}_ExplanationList_deep.pkl'), 'rb') as file:
                        explanation_list = pickle.load(file)

                    shap_val_list = []
                    for explain in explanation_list:
                        if key == 'MLPClassifier':
                            shap_val_list.append(explain.values[:, :, 0])
                        else:
                            shap_val_list.append(explain.values[:, :, 1])

                    avg_shap_val = np.mean(shap_val_list, axis=0)

                    shap.summary_plot(avg_shap_val, test_df, show=False, plot_type="dot", max_display=10, feature_names= test_df.columns)
                    plt.savefig(os.path.join(single_result_path, f'{key}_beeswarmDeep.png'), dpi=500)
                    plt.close()
                    shap.summary_plot(avg_shap_val, test_df, show=False, plot_type="bar", max_display=10, feature_names= test_df.columns)
                    plt.savefig(os.path.join(single_result_path, f'{key}_barplotDeep.png'), dpi=500)
                    plt.close()




        #DEG Data
            else:
                key = os.path.split(path_to_ML_model)[-1]

                test = test_df.drop(columns=['case_id'])
                features = test.columns.to_list()
                X_log = np.log1p(test)
                test = StandardScaler().fit_transform(X_log)

                explanation_list = []

                if not (load_explain):
                    classifiers_list = classifiers_dict[key]

                    for model in classifiers_list:
                        if model_name == 'MLPClassifier':
                            explainer = shap.GradientExplainer(model, test)
                            explanation_list.append(explainer(test))
                        else:
                            explainer = shap.TreeExplainer(model)
                            explanation_list.append(explainer(test))

                    print("Explanation created!")


                    with open(os.path.join(single_result_path, f'{key}_ExplanationList.pkl'), 'wb') as file:
                        pickle.dump(explanation_list, file)


                else:
                    print("Loading Explainers")

                    with open(os.path.join(single_result_path, f'{key}_ExplanationList.pkl'), 'rb') as file:
                        explanation_list = pickle.load(file)
                    shap_val_list = []
                    expected_val = []

                    for sample_id in sample_list:

                        if local_XAI: #retieve expected value from models
                            XAI_sample_ID, sample_index = get_sample(id_pat=sample_id, df=test_df)

                        for key, values in classifiers_dict.items():
                            expected_val.append([classifier.predict(test[sample_index].reshape(1,-1)) for classifier in values])
                        expected_value = np.mean(expected_val[0])

                        for explain in explanation_list:
                            if key == 'MLPClassifier':
                                shap_val_list.append(explain.values[:, :, 0])
                            elif key == 'XGBClassifier':
                                    shap_val_list.append(explain.values)
                            else: shap_val_list.append(explain.values[:, :, 1])



                        avg_shap_val = np.mean(shap_val_list, axis=0)



                        if local_XAI:

                            shap_values_sample = avg_shap_val[sample_index, :]
                            features_sample = np.array(features)
                            top_10_indices = np.argsort(np.abs(shap_values_sample))[-10:]
                            top_10_shap_values = shap_values_sample[top_10_indices]
                            top_10_features = features_sample[top_10_indices]
                            plt.rcParams.update({'font.size': 22})

                            #plt.rcParams.update({'font.size': 22})
                            shap.force_plot(expected_value, top_10_shap_values, feature_names=top_10_features,
                                            show=False, matplotlib=True, figsize=(12,6), text_rotation=90)
                            plt.subplots_adjust(top=0.7)
                            #plt.show()
                            processed = f"{sample_index}_{sample_id}_{gene}_{model_name}"

                            #shap.force_plot(expected_value, avg_shap_val[sample_index,:], show=False, matplotlib=True, feature_names=features)

                            plt.savefig(os.path.join(single_result_path, f'{key}_Decision_plot_{sample_id}.png'), dpi=500)
                            plt.close()

                        else:
                            plt.rcParams.update({'font.size': 16})
                            shap.summary_plot(avg_shap_val, test, show=False, feature_names=features, plot_type="dot", max_display=10)
                            plt.savefig(os.path.join(single_result_path, f'{key}_beeswarm.png'), dpi=500)
                            plt.close()
                            plt.rcParams.update({'font.size': 16})
                            shap.summary_plot(avg_shap_val, test, show=False, feature_names=features, plot_type="bar", max_display=10)
                            plt.savefig(os.path.join(single_result_path, f'{key}_barplot.png'), dpi=500)
                            plt.close()



