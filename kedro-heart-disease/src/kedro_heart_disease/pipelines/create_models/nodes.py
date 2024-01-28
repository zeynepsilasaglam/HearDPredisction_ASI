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
from sklearn.tree import DecisionTreeClassifier

from kedro_heart_disease.pipelines.kedro_heart_disease.nodes import model_score, split_data
import optuna

rf_model = RandomForestClassifier()
gnb_model = GaussianNB()
dt_model = DecisionTreeClassifier()

def optimize_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)
    print("Shape",X_train.shape[0])

    def objective_rf(trial: optuna.Trial):
        n_estim = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        rf_model.set_params(n_estimators=n_estim, max_depth=max_depth)
        rf_model.fit(X_train, y_train)
        return model_score(rf_model)

    def objective_dt(trial: optuna.Trial):
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        dt_model.set_params(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        dt_model.fit(X_train, y_train)
        return model_score(dt_model)
    
    study_rf = optuna.create_study(
        study_name="random-forest",
        direction = optuna.study.StudyDirection.MAXIMIZE
    )

    study_dt = optuna.create_study(
        study_name="decision-tree",
        direction = optuna.study.StudyDirection.MINIMIZE
    )

    study_rf.optimize(func=objective_rf, n_trials=100, show_progress_bar=True)
    study_dt.optimize(func=objective_dt, n_trials=100, show_progress_bar=True)

    rf_model.set_params(**study_rf.best_params)
    dt_model.set_params(**study_dt.best_params)

    print("params for rf: ", study_rf.best_params)
    print("params for dt: ", study_dt.best_params)

    return rf_model, gnb_model, dt_model



def fit_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)

    rf_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/rf_model.pkl")
    model.save(rf_model)

    gnb_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/gnb_model.pkl")
    model.save(gnb_model)

    dt_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/dt_model.pkl")
    model.save(dt_model)
    




