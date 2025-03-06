import pandas as pd
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

import os
from sklearn.metrics import roc_auc_score, average_precision_score

def training_models(X_train, y_train, save_path):


    np.random.seed(42)
    random_state = 42

    rf_classifier = RandomForestClassifier(random_state=random_state)
    xgb_classifier = xgb.XGBClassifier(random_state=random_state)

    classifiers = [rf_classifier, xgb_classifier] #Linear regression, random forest, XGB and MLP

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits)
    classifiers_dict = {}

    #param grids for fine tuning



    for classifier in classifiers:  #for each classifier
        classifier_name = classifier.__class__.__name__
        print(f"Evaluating classifier: {classifier_name} with {n_splits}-fold CV...")
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        classifiers_dict[classifier_name] = []
        performance_dict = {"Fold": [], "AUROC": [], "AUPRC": []}



        for fold, (train, val) in enumerate(cv.split(X_train, y_train)): #train with 10 fold CV the best estimator
            print("Training model on {}-fold".format(fold))
            classifier_path = os.path.join(save_path, classifier_name, f"model_{fold}.ckpt")
            classifier = joblib.load(classifier_path)

            if isinstance(X_train, pd.DataFrame):
                X_train_f, y_train_f = X_train.iloc[train].values, y_train.iloc[train].values
                X_val, y_val = X_train.iloc[val].values, y_train.iloc[val].values
                #classifier.fit(X_train_f, y_train_f.ravel())  # train

            else:
                X_train_f, y_train_f = X_train[train], y_train[train]
                X_val, y_val = X_train[val], y_train[val]
                #classifier.fit(X_train_f, y_train_f)  # train


            y_val_pred = classifier.predict_proba(X_val)[:,1]
            auroc = roc_auc_score(y_val, y_val_pred)
            auprc = average_precision_score(y_val, y_val_pred)
            performance_dict["FOLD"].append(fold)
            performance_dict["AUROC"].append(auroc)
            performance_dict["AUPRC"].append(auprc)

        performance = pd.DataFrame(performance_dict)
        avg_auroc = performance["AUROC"].mean()
        avg_auprc = performance["AUPRC"].mean()

        classifiers_dict[classifier_name].append([avg_auroc, avg_auprc])
    avg_perf = pd.DataFrame(classifiers_dict)
    avg_perf.to_csv(os.path.join())




