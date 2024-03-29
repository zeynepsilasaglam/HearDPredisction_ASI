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
from sklearn.metrics import f1_score, mean_squared_error
from enum import Enum
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from kedro_datasets.pickle import PickleDataSet
from sklearn.tree import DecisionTreeClassifier
import logging

class ModelNames(Enum):
    GAUSSIAN_NB = "GaussianNB"
    RANDOM_FOREST = "RandomForestClassifier"
    DT_MODEL = "DecisionTreeClassifier"

rf_model_file = "data/06_models/rf_model.pkl"
gnb_model_file = "data/06_models/gnb_model.pkl"
dt_model_file = "data/06_models/dt_model.pkl"

# rf_model = RandomForestClassifier()
# gnb_model = GaussianNB()
# dt_model = DecisionTreeClassifier()

rf_model_pkl = PickleDataSet(filepath=rf_model_file) 
gnb_model_pkl = PickleDataSet(filepath=gnb_model_file)
dt_mode_pkl = PickleDataSet(filepath=dt_model_file)

if os.path.exists(rf_model_file):
    rf_model = rf_model_pkl.load()
if os.path.exists(gnb_model_file):
    gnb_model = gnb_model_pkl.load()
if os.path.exists(dt_model_file):
    dt_model = dt_mode_pkl.load()


def check_model(model_name: str):
    try:
        return {
            ModelNames.RANDOM_FOREST: rf_model,
            ModelNames.GAUSSIAN_NB: gnb_model,
            ModelNames.DT_MODEL: dt_model,
        }[ModelNames(model_name)]
    except ValueError:
        raise ValueError("Unknown algorithm name")     
            

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
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X_test.columns = columns
    y_pred = model.predict(X_test)
    #print(X_test)
    score = ''
    if type(model).__name__ == ModelNames.RANDOM_FOREST.value:
        score = f1_score(y_test, y_pred, average='weighted')
    if type(model).__name__ == ModelNames.GAUSSIAN_NB.value:
        score = mean_squared_error(y_test, y_pred)
    if type(model).__name__ == ModelNames.DT_MODEL.value:
        score = f1_score(y_test, y_pred)

    return score


def train(model, X_train: pd.DataFrame, y_train: pd.DataFrame):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X_train.columns = columns
    y_train.columns = ["target"]
    current_model = model.fit(X_train, y_train)
    
    if type(model).__name__ == ModelNames.RANDOM_FOREST.value:
        rf_model_pkl.save(model)
    if type(model).__name__ == ModelNames.GAUSSIAN_NB.value:
        gnb_model_pkl.save(model)
    if type(model).__name__ == ModelNames.DT_MODEL.value:
        dt_mode_pkl.save(model)


    return current_model
    

def predict(model, data: pd.DataFrame) :
    logging.warning(rf_model.get_params)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    data.columns = columns
    prediction = model.predict(data)
    return prediction
