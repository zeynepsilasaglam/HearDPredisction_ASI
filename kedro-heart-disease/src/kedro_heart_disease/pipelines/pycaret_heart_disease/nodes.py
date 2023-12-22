"""
This is a boilerplate pipeline 'pycaret_heart_disease'
generated using Kedro 0.19.1
"""

import logging
from typing import Any, Dict, Tuple
from pycaret.datasets import get_data
from pycaret.classification import *
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef


def find_best_model(data: pd.DataFrame, target) -> Tuple[Any, Any]:
    model_setup = setup(data, target=target, session_id=111)
    best_model = model_setup.compare_models(turbo=False)
    return best_model, model_setup

def predict_by_best_model(best_model, data: pd.DataFrame, model_setup):
    prediction = model_setup.predict_model(best_model, data=data, raw_score=True)
    print(prediction)
    return prediction

def scoring(best_model, prediction):
    y_true = prediction["target"]
    y_pred = prediction["prediction_label"]
    print("Accuracy: {0:.3f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {0:.3f}".format(precision_score(y_true, y_pred)))
    print("Recall: {0:.3f}".format(recall_score(y_true, y_pred)))
    print("F1-score: {0:.3f}".format(f1_score(y_true, y_pred)))
    print("MCC: {0:.3f}".format(matthews_corrcoef(y_true, y_pred)))
