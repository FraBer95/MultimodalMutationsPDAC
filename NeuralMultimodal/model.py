import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, roc_auc_score, average_precision_score
import os
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping


def build_mlp_model(input_dim_img, input_dim_ts, hidden_layer_sizes=(100, 50, 25), activation='relu', alpha=0.001, learning_rate=0.001):
    input1 = Input(shape=(input_dim_img,), name="Pathomics")
    x1 = Dense(256, activation="relu")(input1)
    #x1 = Dense(32, activation="relu")(x1)
    x1 = Dropout(0.3)(x1)


    input2 = Input(shape=(input_dim_ts,), name="transcriptomics")
    x2 = Dense(256, activation="relu")(input2)
    #x2 = Dense(32, activation="relu")(x2)
    x2 = Dropout(0.3)(x2)


    merged = Concatenate()([x1, x2])


    x = Dense(256, activation="relu")(merged)


    # Output per la classificazione
    output = Dense(1, activation="sigmoid", name="classification_output")(x)


    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model



def training_keras_model(X_trainIm, X_trainTs, y, save_path, n_splits=10, gene=""):
    np.random.seed(42)

    input_dimIm = X_trainIm.shape[1]
    input_dimTs = X_trainTs.shape[1]

    cv = StratifiedKFold(n_splits=n_splits)

    classifiers_dict = {"MLP": []}
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    performance_list = []
    performance_dict = {"Fold": [], "AUROC": [], "AUPRC": []}
    for fold, (train, val) in enumerate(cv.split(X_trainIm, y)):

        print(f"Training fold {fold + 1}/{n_splits}")

        X_train_img, X_train_rna, y_train_f = X_trainIm.iloc[train].values, X_trainTs.iloc[train].values, y.iloc[train].values
        X_val_img, X_val_rna, y_val_f = X_trainIm.iloc[val].values, X_trainTs.iloc[val].values, y.iloc[val].values

        model = build_mlp_model(input_dimIm, input_dimTs)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=60,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )

        history = model.fit(
            [X_train_img, X_train_rna],
            y_train_f,  # Output
            validation_data=([X_val_img, X_val_rna], y_val_f),  # Dati di validazione
            epochs=200,
            batch_size=64,
            verbose=1,
            callbacks=[early_stopping]
        )

        classifiers_dict["MLP"].append(model)
        ckpt_path = os.path.join(save_path, 'ckpt')
        os.makedirs(ckpt_path, exist_ok=True)
        model.save(os.path.join(ckpt_path, f"mlp_keras_model_fold_{fold}.h5"))


        y_val_pred_proba = model.predict([X_val_img, X_val_rna]).ravel()


        auroc = roc_auc_score(y_val_f, y_val_pred_proba)
        auprc = average_precision_score(y_val_f, y_val_pred_proba)

        classifiers_dict["MLP"].append(model)
        # os.makedirs(save_path, exist_ok=True)
        # model.save(os.path.join(save_path, f"mlp_keras_model_fold_{fold}.h5"))

        # Valutazione del modello e calcolo della curva ROC

        performance_dict["Fold"].append(fold)
        performance_dict["AUROC"].append(auroc)
        performance_dict["AUPRC"].append(auprc)

    performance = pd.DataFrame(performance_dict)
    avg_auroc = performance["AUROC"].mean()
    avg_auprc = performance["AUPRC"].mean()

    performance_list.append({"Model": "MLP", "AUROC": avg_auroc, "AUPRC": avg_auprc})
    # classifiers_dict["MLP"].append([avg_auroc, avg_auprc])

    avg_perf = pd.DataFrame(performance_list)
    # avg_perf["Model"] = classifier_name
    # avg_perf = avg_perf[["Model", "AUROC", "AUPRC"]]
    avg_perf.to_csv(os.path.join(save_path, 'MLPsubject_metrics.csv'))

    return classifiers_dict

