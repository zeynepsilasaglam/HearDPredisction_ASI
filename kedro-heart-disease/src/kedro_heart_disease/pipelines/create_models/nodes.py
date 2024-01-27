"""
This is a boilerplate pipeline 'create_models'
generated using Kedro 0.19.1
"""
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from kedro_datasets.pickle import PickleDataSet
from sklearn.linear_model import LogisticRegression

from kedro_heart_disease.pipelines.kedro_heart_disease.nodes import model_score, split_data

import optuna


rf_model = RandomForestClassifier()
gnb_model = GaussianNB()
lr_model = LogisticRegression()

def optimize_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)
    print("Shape",X_train.shape[0])

    def objective_rf(trial: optuna.Trial):
        n_estim = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        rf_model.set_params(n_estimators=n_estim, max_depth=max_depth)
        rf_model.fit(X_train, y_train)
        return model_score(rf_model)
    
    def objective_lr(trial: optuna.Trial):
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        lr_model.set_params(C=C, max_iter=max_iter)
        lr_model.fit(X_train, y_train)
        return model_score(lr_model)

    study_rf = optuna.create_study(
        study_name="random-forest",
        direction = optuna.study.StudyDirection.MAXIMIZE
    )

    study_lr = optuna.create_study(
        study_name="log reg",
        direction = optuna.study.StudyDirection.MAXIMIZE
    )

    study_rf.optimize(func=objective_rf, n_trials=100, show_progress_bar=True)
    rf_model.set_params(**study_rf.best_params)

    study_lr.optimize(func=objective_lr, n_trials=100, show_progress_bar=True)
    lr_model.set_params(**study_lr.best_params)

    print("params for rf: ", study_rf.best_params)
    print("params for lr: ", study_lr.best_params)

    print(lr_model.get_params())

    return rf_model, lr_model, gnb_model



def fit_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)

    rf_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/rf_model.pkl")
    model.save(rf_model)

    gnb_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/gnb_model.pkl")
    model.save(gnb_model)

    lr_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/lr_model.pkl")
    model.save(lr_model)
    




