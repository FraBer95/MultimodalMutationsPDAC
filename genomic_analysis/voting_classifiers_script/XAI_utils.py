import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras import Model, Input



def rename_genes(conversion_table, df, columns):

    for gene in tqdm(columns):
        try:
            new_name = conversion_table[conversion_table['gene_id'] == gene].iloc[:, 2].iloc[0]
            if new_name == 'No conversion':
                new_name = conversion_table[conversion_table['gene_id'] == gene].iloc[:, 0].iloc[0]
            df.rename(columns={gene:new_name}, inplace=True)
        except:
            print(f"Gene name not found: {gene}")
            pass
    #df.columns = df.columns.astype(str)

    return df

def get_classifiers(classifiers_path):

    class_dict = {}
    classifier = os.path.split(classifiers_path)[-1]
    class_list  = []
    models = os.listdir(classifiers_path)

    for model in models:
        if classifier != 'MLPClassifier':
            class_list.append(joblib.load(os.path.join(classifiers_path, model)))
        else:
            class_list.append(load_model(os.path.join(classifiers_path, model)))

    class_dict[classifier] = class_list
    return class_dict


def predict(encoder, classifier_dict, data):

    encoded_data = encoder.predict(data)
    y_score = np.array([c.predict_proba(encoded_data) for c in classifier_dict])
    y_score_means = np.mean(y_score, axis=0)
    #y_pred = np.argmax(y_score_means, axis=1)

    return y_score_means


def predict_DEG(classifier_dict, data):

    y_score = np.array([c.predict_proba(data) for c in classifier_dict])
    y_score_means = np.mean(y_score, axis=0)
    #y_pred = np.argmax(y_score_means, axis=1)

    return y_score_means

def get_sample(id_pat, id_list=None, df=None):

    if 'case_id' in df.columns and id_list is None:
        id_list = df['case_id'].tolist()
        index = id_list.index(id_pat)

    else: index = id_list[id_list == id_pat].index[0]
    sample = df.iloc[index, :]

    return sample, index

def get_top_features(shap_values, k):
    shap_values_abs = np.abs(shap_values)
    top_k_indices = np.argsort(shap_values_abs)[-k:]

    return top_k_indices


def create_models(encoder, keras_dict_model):

    new_models_list = []
    new_dict = {'MLPClassifier': []}
    for mlp_model in  next(iter(keras_dict_model.values())):

        input_shape = encoder.input_shape[1:]
        input_layer = Input(shape=input_shape)

        encoded_output = encoder(input_layer)
        mlp_output = mlp_model(encoded_output)

        new_model = Model(inputs=input_layer, outputs=mlp_output)

        new_models_list.append(new_model)



    return  new_models_list


