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


def split_data(data: pd.DataFrame):
    data_train = data.sample(frac=0.7, random_state=42)
    data_test = data.drop(data_train.index)
    X_train = data_train.drop(columns="target")
    X_test = data_test.drop(columns="target")
    y_train = data_train["target"]
    y_test = data_test["target"]
    return X_train, X_test, y_train, y_test

def create_models(data: pd.DataFrame)-> Tuple[Any, Any, Any, Any]:
    X_train, X_test, y_train, y_test = split_data(data)

    rand_for_model = RandomForestClassifier()
    rand_for_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/rand_for_model.pkl")
    model.save(rand_for_model)

    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/knn_model.pkl")
    model.save(knn_model)
    
    gauss_model = GaussianNB()
    gauss_model.fit(X_train, y_train)
    model = PickleDataSet(filepath="data/06_models/gauss_model.pkl")
    model.save(gauss_model)

    current_model = rand_for_model
    return rand_for_model, knn_model, gauss_model, current_model
