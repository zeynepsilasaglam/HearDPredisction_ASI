"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

import logging
from symbol import parameters
from turtle import mode
from typing import Any, Dict, Tuple
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def split_data(
    data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample(frac=0.7, random_state=42)
    data_test = data.drop(data_train.index)
    X_train = data_train.drop(columns="target")
    X_test = data_test.drop(columns="target")
    y_train = data_train["target"]
    y_test = data_test["target"]

    return X_train, X_test, y_train, y_test


def models(data: pd.DataFrame)-> Tuple[Any, Any, Any, Any]:
    
    X_train, X_test, y_train, y_test = split_data(data)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    return random_forest, knn, logistic_regression, gaussian
    

def make_predictions_all_models(random_forest, knn, logistic_regression, gaussian, X_test: pd.DataFrame
) -> pd.Series:
    """Uses all models to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """
    random_forest_predict = random_forest.predict(X_test)
    knn_predict = knn.predict(X_test)
    logistic_regression_predict = logistic_regression.predict(X_test)
    gaussian_predict = gaussian.predict(X_test)

    return random_forest_predict, knn_predict, logistic_regression_predict, gaussian_predict


def accuracy(model,  y_test: pd.Series) -> float:
    return (model == y_test).sum() / len(y_test)
    

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
    result = True if prediction[0] == 1 else False
    logger = logging.getLogger(__name__)
    logger.info("Does person have heart disease %s", str(result))
