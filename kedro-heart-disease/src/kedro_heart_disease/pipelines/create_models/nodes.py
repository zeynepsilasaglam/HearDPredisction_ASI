"""
This is a boilerplate pipeline 'create_models'
generated using Kedro 0.19.1
"""
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from kedro_datasets.pickle import PickleDataSet

from kedro_heart_disease.pipelines.kedro_heart_disease.nodes import model_score, split_data

import optuna


rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
gnb_model = GaussianNB()

def optimize_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)
    print("Shape",X_train.shape[0])

    def objective_rf(trial: optuna.Trial):
        n_estim = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        rf_model.set_params(n_estimators=n_estim, max_depth=max_depth)
        rf_model.fit(X_train, y_train)
        return model_score(rf_model)
    
    def objective_knn(trial: optuna.Trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        knn_model.set_params(n_neighbors=n_neighbors, weights=weights)
        knn_model.fit(X_train, y_train)
        return model_score(knn_model)

    study_rf = optuna.create_study(
        study_name="random-forest",
        direction = optuna.study.StudyDirection.MAXIMIZE
    )

    study_knn = optuna.create_study(
        study_name="knn",
        direction = optuna.study.StudyDirection.MAXIMIZE
    )

    study_rf.optimize(func=objective_rf, n_trials=100, show_progress_bar=True)
    rf_model.set_params(**study_rf.best_params)

    study_knn.optimize(func=objective_knn, n_trials=100, show_progress_bar=True)
    knn_model.set_params(**study_knn.best_params)

    print("params for rf: ", study_rf.best_params)
    print("params for knn: ", study_knn.best_params)

    knn_model.predict(X_test)
    print(model_score(knn_model))

    return rf_model, knn_model, gnb_model



def fit_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)

    rf_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/rf_model.pkl")
    model.save(rf_model)

    knn_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/knn_model.pkl")
    model.save(knn_model)

    gnb_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/gnb_model.pkl")
    model.save(gnb_model)




