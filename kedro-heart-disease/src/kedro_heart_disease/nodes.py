"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from importlib.machinery import ModuleSpec
import logging
#from symbol import parameters
#from turtle import mode
from typing import Any, Dict, Tuple
#from xmlrpc.client import Boolean
from enum import Enum


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Models(str, Enum):
    rand_for = "RandomForestClassifier",
    knn = "KNeighborsClassifier",
    log_reg = "LogisticRegression",
    gauss = "GaussianNB"

rand_for_model = RandomForestClassifier()
log_reg_model = LogisticRegression() 
knn_model = KNeighborsClassifier()
gauss_model = GaussianNB()
current_model = rand_for_model

def check_model(model: str):
    for models in Models:
        if models.value == model == "RandomForestClassifier":
            return rand_for_model
        if models.value == model == "LogisticRegression":
            return log_reg_model
        if models.value == model == "KNeighborsClassifier":
            return knn_model
        if models.value == model == "GaussianNB":
            return gauss_model        
            

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


def train_one(model, data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)
    current_model.fit(X_train, y_train)
    return current_model


def train(model, data: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(data)
    current_model.fit(X_train, y_train)
    return current_model
    

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def accuracy(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Classification Report: %s", cm)
    

def report_accuracy(random_forest_predict: pd.Series, knn_predict: pd.Series, logistic_regression_predict: pd.Series, gaussian_predict: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    logger = logging.getLogger(__name__)
    logger.info("Model random_forest has accuracy of %.3f on test data.", accuracy(random_forest_predict, y_test))
    logger.info("Model knn_predict has accuracy of %.3f on test data.", accuracy(knn_predict, y_test))
    logger.info("Model logistic_regression has accuracy of %.3f on test data.", accuracy(logistic_regression_predict, y_test))
    logger.info("Model gaussian has accuracy of %.3f on test data.", accuracy(gaussian_predict, y_test))



def predict(model, parameters: Dict[str, Any]) :
    array = np.array(parameters["test_data"]). reshape(1, 13)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = pd.DataFrame(data=array, columns=columns)

    prediction = model.predict(df)
    return prediction[0]
