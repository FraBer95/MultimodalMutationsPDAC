import pandas as pd
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

import os

def training_models(X_train, y_train, save_path):


    np.random.seed(42)
    tuning = True
    random_state = 42

    rf_classifier = RandomForestClassifier(random_state=random_state)
    xgb_classifier = xgb.XGBClassifier(random_state=random_state)

    classifiers = [rf_classifier, xgb_classifier] #Linear regression, random forest, XGB and MLP

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits)
    classifiers_dict = {}

    #param grids for fine tuning

    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 25, 50, 75],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    param_grid_xgb = {
        'n_estimators': [100, 200, 500],
        'max_depth': [15, 25, 50],
        'learning_rate': [0.01, 0.1, 0.001],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    for classifier in classifiers:  #for each classifier
        classifier_name = classifier.__class__.__name__
        print(f"Training classifier: {classifier_name} with {n_splits}-fold CV...")
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        classifiers_dict[classifier_name] = []


        if tuning: #if true do the fine tuning according model param grid
            print("Tuning hyperparameters...")
            if classifier_name == rf_classifier.__class__.__name__:
                grid = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5, scoring='roc_auc')
            elif classifier_name == xgb_classifier.__class__.__name__:
                grid = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, cv=5, scoring='roc_auc')
            grid.fit(X_train, y_train)
            classifier = grid.best_estimator_ #take the best estimator with hyper-params
            print(f"Best parameters for {classifier_name}: {grid.best_params_}")

        fig, ax = plt.subplots(figsize=(6, 6))

        for fold, (train, val) in enumerate(cv.split(X_train, y_train)): #train with 10 fold CV the best estimator
            print("Training model on {}-fold".format(fold))
            if isinstance(X_train, pd.DataFrame):
                X_train_f, y_train_f = X_train.iloc[train].values, y_train.iloc[train].values
                X_val, y_val = X_train.iloc[val].values, y_train.iloc[val].values
                classifier.fit(X_train_f, y_train_f.ravel())  # train

            else:
                X_train_f, y_train_f = X_train[train], y_train[train]
                X_val, y_val = X_train[val], y_train[val]
                classifier.fit(X_train_f, y_train_f)  # train



            classifiers_dict[classifier_name].append(classifier) #append the trained classifier model into a dict having classifier name as key and values the trained models
            classifier_path = os.path.join(save_path, classifier_name)
            os.makedirs(classifier_path, exist_ok=True)
            joblib.dump(classifier, os.path.join(classifier_path, f"model_{fold}.ckpt"))

            #for each fold compute and plot the ROC Curves with CI
            viz = RocCurveDisplay.from_estimator(
                classifier,
                X_val,
                y_val,
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
            )

            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability \n for {classifier_name}"
        )
        ax.axis("square")
        ax.legend(loc="lower right")

        #plt.savefig(os.path.join(save_path, classifier_name+"_crossVal.png"))

        plt.show()
        # plt.close()




    return classifiers_dict #return the dict with trained classifier models
