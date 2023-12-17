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
oi = DataCatalog()
#io = DataCatalog()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Models(str, Enum):
    rand_for = "RandomForestClassifier",
    knn = "KNeighborsClassifier",
    log_reg = "LogisticRegression",
    gauss = "GaussianNB"

rand_for_model = RandomForestClassifier()
log_reg_model = LogisticRegression(max_iter=1000) 
knn_model = KNeighborsClassifier()
gauss_model = GaussianNB()
current_model = rand_for_model
score = {}

def check_model(model: str):
    if model == "RandomForestClassifier":
        return rand_for_model
    elif model == "LogisticRegression":
        return log_reg_model
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


def models(data: pd.DataFrame)-> Tuple[Any, Any, Any, Any]:

    X_train, X_test, y_train, y_test = split_data(data)
    rand_for_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    log_reg_model.fit(X_train, y_train)
    gauss_model.fit(X_train, y_train)
    current_model = rand_for_model
    return rand_for_model, knn_model, log_reg_model, gauss_model, current_model


def model_score(model):
    dr = oi.load("heart_disease_data")
    print(dr)
    data = io.load("heart_disease_data")
    X_train, X_test, y_train, y_test = split_data(data)
    y_train = y_train.values
    score[type(model).__name__] = model.score(X_test, y_test)
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
