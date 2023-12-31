"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from importlib.machinery import ModuleSpec
import logging
from typing import Any, Dict, Tuple
from enum import Enum
from kedro_datasets.pandas import CSVDataset
from kedro.io import DataCatalog
io = DataCatalog(datasets={
                  "heart_disease_data": CSVDataset(filepath="data/01_raw/heart.csv")
                  })

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score

class Models(str, Enum):
    rand_for = "RandomForestClassifier",
    knn = "KNeighborsClassifier",
    gauss = "GaussianNB"

rand_for_model = RandomForestClassifier()
knn_model = KNeighborsClassifier(n_neighbors=1)
gauss_model = GaussianNB()
current_model = rand_for_model

def check_model(model: str):
    if model == "RandomForestClassifier":
        return rand_for_model
    elif model == "KNeighborsClassifier":
        return knn_model
    elif model == "GaussianNB":
        return gauss_model  
    else:
        raise ValueError("Unknown algorithm name")      
            

def get_model(model: str):
    current_model = check_model(model)
    return current_model

def get_current_model():
    return current_model 


def split_data(data: pd.DataFrame):
    data_train = data.sample(frac=0.7, random_state=42)
    data_test = data.drop(data_train.index)
    X_train = data_train.drop(columns="target")
    X_test = data_test.drop(columns="target")
    y_train = data_train["target"]
    y_test = data_test["target"]
    return X_train, X_test, y_train, y_test


def optimize_(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)

    def objective_rf(trial: optuna.Trial):
        n_estim = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        rand_for_model.set_params(n_estimators=n_estim, max_depth=max_depth)
        rand_for_model.fit(X_train, y_train)
        return model_score(rand_for_model)
    
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
    rand_for_model.set_params(**study_rf.best_params)

    study_knn.optimize(func=objective_knn, n_trials=100, show_progress_bar=True)
    knn_model.set_params(**study_knn.best_params)

    print("weights for rf: ", study_rf.best_params)
    print("weights for knn: ", study_knn.best_params)


def models(data: pd.DataFrame)-> Tuple[Any, Any, Any, Any]:
    X_train, X_test, y_train, y_test = split_data(data)
    rand_for_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    gauss_model.fit(X_train, y_train)
    current_model = rand_for_model
    return rand_for_model, knn_model, gauss_model, current_model


def model_score(model):
    data = io.load("heart_disease_data")
    X_train, X_test, y_train, y_test = split_data(data)
    y_train = y_train.values
    y_pred = model.predict(X_test)
    score = ''
    if type(model).__name__ == "RandomForestClassifier":
        score = f1_score(y_test, y_pred, average='weighted')
    if type(model).__name__ == "GaussianNB":
        score = mean_squared_error(y_test, y_pred)
    if type(model).__name__ == "KNeighborsClassifier":
        score = balanced_accuracy_score(y_test, y_pred)
    return score


def train(model, X_train: pd.DataFrame, y_train: pd.DataFrame):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X_train.columns = columns
    y_train.columns = ["target"]
    current_model = model.fit(X_train, y_train)
    

def predict(model, data: pd.DataFrame) :
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    data.columns = columns
    prediction = model.predict(data)
    return prediction
