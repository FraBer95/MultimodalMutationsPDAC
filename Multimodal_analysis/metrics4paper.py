import pandas as pd
import os
import openpyxl

def mapping_column(col_name, model):


    if model == 'CONCH':
        mod = 'C'
    elif model == 'Resnet':
        mod = 'RN'
    elif model == 'UNI':
        mod = 'U'
    else: print("Model not found")


    tokens = col_name.split("_")

    if '64' in tokens:
        prep = "AE64"
    elif '128' in tokens:
        prep = "AE128"
    elif '256' in tokens:
        prep = "AE256"
    else: prep = "DS" # DeseqData


    if 'RF' in tokens:
        classifier = 'R'
    elif 'XGBClassifier' in tokens:
        classifier = 'X'
    elif 'MLP' in tokens:
        classifier = 'M'

    if 'big' in tokens: size = 'L'
    else: size = 'S'


    new_col_name = prep+classifier+"+"+mod+"+"+size

    return new_col_name









if __name__ == '__main__':

    metrics_path = r'E:\Users\Berloco\PycharmProjects\CLAM\Multimodal_analysis\multimodal_metrics_new'

    genes = ['SMAD4', 'TP53', 'CDKN2A', 'KRAS']
    save_folder = r'C:\Users\Berloco\\Desktop\Paper_result\multimodal_recomputed2210_new'
    os.makedirs(save_folder, exist_ok=True)

    resnet_new_order = [
    "DSM+RN+L", "DSR+RN+L", "DSX+RN+L", "AE64M+RN+L", "AE64R+RN+L", "AE64X+RN+L",
    "AE128M+RN+L", "AE128R+RN+L", "AE128X+RN+L", "AE256M+RN+L", "AE256R+RN+L", "AE256X+RN+L",
    "DSM+RN+S", "DSR+RN+S", "DSX+RN+S", "AE64M+RN+S", "AE64R+RN+S", "AE64X+RN+S",
    "AE128M+RN+S", "AE128R+RN+S", "AE128X+RN+S", "AE256M+RN+S", "AE256R+RN+S", "AE256X+RN+S"
    ]

    conch_new_order = [
        "DSM+C+L", "DSR+C+L", "DSX+C+L", "AE64M+C+L", "AE64R+C+L", "AE64X+C+L",
        "AE128M+C+L", "AE128R+C+L", "AE128X+C+L", "AE256M+C+L", "AE256R+C+L", "AE256X+C+L",
        "DSM+C+S", "DSR+C+S", "DSX+C+S", "AE64M+C+S", "AE64R+C+S", "AE64X+C+S",
        "AE128M+C+S", "AE128R+C+S", "AE128X+C+S", "AE256M+C+S", "AE256R+C+S", "AE256X+C+S"
    ]

    uni_new_order = [
        "DSM+U+L", "DSR+U+L", "DSX+U+L", "AE64M+U+L", "AE64R+U+L", "AE64X+U+L",
        "AE128M+U+L", "AE128R+U+L", "AE128X+U+L", "AE256M+U+L", "AE256R+U+L", "AE256X+U+L",
        "DSM+U+S", "DSR+U+S", "DSX+U+S", "AE64M+U+S", "AE64R+U+S", "AE64X+U+S",
        "AE128M+U+S", "AE128R+U+S", "AE128X+U+S", "AE256M+U+S", "AE256R+U+S", "AE256X+U+S"
    ]



    for gene in genes:
        os.makedirs(os.path.join(save_folder, gene), exist_ok=True)
        save_folder_gene = os.path.join(save_folder, gene)

        df_multi = pd.read_csv(os.path.join(metrics_path, gene, 'metrics.csv'))
        df_T = df_multi.T
        df_T.columns = df_multi.iloc[:, 0]
        df_T.drop(index="Unnamed: 0", inplace=True)
        df_T.drop(index="std on mean_p1", inplace=True)

        models = ['CONCH', 'Resnet', 'UNI']


        for model in models:
            df_model =  df_T.filter(like=model)
            if model == 'CONCH':
                new_order = conch_new_order
            elif model == 'Resnet':
                new_order = resnet_new_order
            elif model == 'UNI':
                new_order = uni_new_order

            for col in df_model.columns:
                new_name = mapping_column(col, model)
                df_model.rename(columns={col: new_name}, inplace=True)
                # tokens = col.str.split('_', expand=True)
                # if (token == 'XGB_Classifier' for token in tokens):
                #     df_model

            try:
                df_ordered = df_model[new_order]
                shape = df_ordered.shape[1]
            except KeyError:
                pass
            df_ordered.to_csv(os.path.join(save_folder_gene, '{}.csv'.format(model)), index=True)

