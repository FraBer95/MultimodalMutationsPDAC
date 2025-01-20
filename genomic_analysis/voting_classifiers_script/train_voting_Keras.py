import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import os


# Funzione per creare il modello MLP in Keras
def build_mlp_model(input_dim, hidden_layer_sizes=(100, 50, 25), activation='relu', alpha=0.001, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Aggiungiamo i layer nascosti specificati
    for units in hidden_layer_sizes:
        model.add(layers.Dense(units=units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(alpha)))

    # Aggiungi layer di output
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compilazione del modello
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['AUC'])

    return model


def training_keras_model(X_train, y_train, save_path, n_splits=10, gene=""):
    np.random.seed(42)

    input_dim = X_train.shape[1]
    cv = StratifiedKFold(n_splits=n_splits)

    # Lista per raccogliere metriche di ogni fold
    classifiers_dict = {}
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    for fold, (train, val) in enumerate(cv.split(X_train, y_train)):
        print(f"Training fold {fold + 1}/{n_splits}")

        # Estrai i dati di train e validazione per questo fold
        X_train_f, y_train_f = X_train.iloc[train].values, y_train.iloc[train].values
        X_val, y_val = X_train.iloc[val].values, y_train.iloc[val].values

        # Crea e addestra il modello
        model = build_mlp_model(input_dim=input_dim, )
        history = model.fit(X_train_f, y_train_f, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, verbose=0)

        # Salva il modello di questo fold
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, f"mlp_keras_model_fold_{fold}.h5"))

        # Valutazione del modello e calcolo della curva ROC
        y_val_pred_proba = model.predict(X_val).ravel()
        fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Visualizza la curva ROC per ogni fold
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"Fold {fold}").plot(ax=ax, alpha=0.3)

        # Interpolazione dei TPR per calcolare la media
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    # Calcola la media e la deviazione standard delle curve ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Visualizzazione della curva ROC media
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = %0.2f $\pm$ %0.2f) for {gene}' % (mean_auc, std_auc),
            lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='Mean ROC curve with variability')
    ax.legend(loc='lower right')

    plt.show()

    return classifiers_dict

