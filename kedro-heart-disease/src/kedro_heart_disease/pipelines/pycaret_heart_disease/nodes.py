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


def find_best_model(data: pd.DataFrame, target) -> Tuple[Any, Any]:
    model_setup = setup(data, target=target, session_id=111)
    best_model = model_setup.compare_models(turbo=False)
    return best_model, model_setup

def predict_by_best_model(best_model, data: pd.DataFrame, model_setup):
    prediction = model_setup.predict_model(best_model, data=data, raw_score=True)
    print(prediction)
    return prediction

def scoring(best_model, prediction):
    print(prediction["prediction_score_0"])