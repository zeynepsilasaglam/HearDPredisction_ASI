"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from importlib.machinery import ModuleSpec
import logging
from typing import Any, Dict
from kedro_datasets.pickle import PickleDataSet
from kedro_datasets.pandas import CSVDataset
from kedro.io import DataCatalog
io = DataCatalog(datasets={
                  "heart_disease_data": CSVDataset(filepath="data/01_raw/heart.csv")
                  })

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from enum import Enum

class ModelNames(Enum):
    KNN_CLASSIFIER = "KNeighborsClassifier"
    RANDOM_FOREST = "RandomForestClassifier"
    GAUSSIAN_NB = "GaussianNB"

#comm
rf_model = PickleDataSet(filepath="data/06_models/rf_model.pkl").load()
knn_model = PickleDataSet(filepath="data/06_models/knn_model.pkl").load()
gnb_model = PickleDataSet(filepath="data/06_models/gnb_model.pkl").load()
current_model = rf_model

def check_model(model_name: str):
    try:
        return {
            ModelNames.RANDOM_FOREST: rf_model,
            ModelNames.KNN_CLASSIFIER: knn_model,
            ModelNames.GAUSSIAN_NB: gnb_model,
        }[ModelNames(model_name)]
    except ValueError:
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

def model_score(model):
    data = io.load("heart_disease_data")
    X_train, X_test, y_train, y_test = split_data(data)
    y_train = y_train.values
    y_pred = model.predict(X_test)
    score = ''
    if type(model).__name__ == ModelNames.RANDOM_FOREST.value:
        score = f1_score(y_test, y_pred, average='weighted')
    if type(model).__name__ == ModelNames.GAUSSIAN_NB.value:
        score = mean_squared_error(y_test, y_pred)
    if type(model).__name__ == ModelNames.KNN_CLASSIFIER.value:
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
