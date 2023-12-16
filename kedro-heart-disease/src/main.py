from fastapi import FastAPI, Query, Body, Depends
from typing_extensions import Annotated# typing_ext for python <= 3.9 
from typing import List, Any, Tuple
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro_heart_disease.nodes import models, predict

app = FastAPI()

class Models(str, Enum):
    rand_for = "Random forest",
    knn = "K-Nearest Neighbors",
    log_reg = "Logistic Regression",
    gauss = "Gaussian Naive Bayes"


#returns list of models 
@app.get("/models")
def greet() -> List[str]: 
    return [model.value for model in Models]

# class InputEntry(BaseModel):
#     column_name: str
#     value: Any


#takes model name as an input and row. Request body sample: [62, 0, 0, 124, 209, 0, 1, 163, 0, 0, 2, 0, 2]
@app.post("/predict")
def predict(
    model_name: Annotated[Models, Query()],
    input: Annotated[List[int], Body()]) -> Any:
    
    result = models(pd.DataFrame(input))
    print(result)

    return 0


@app.post("/train")
def train(model_name: Annotated[Models, Query()],
    input: Annotated[List[int], Body()],
    expected_output: [List[Any], Body()]) -> Any:

    return 0

