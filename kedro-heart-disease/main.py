from fastapi import FastAPI, Query, Body, Depends
from typing_extensions import Annotated# typing_ext for python <= 3.9 
from typing import List, Any
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from src.kedro_heart_disease.nodes import train, predict, check_model
import src.kedro_heart_disease.nodes

app = FastAPI()

class Models(str, Enum):
    rand_for = "RandomForestClassifier",
    knn = "KNeighborsClassifier",
    log_reg = "LogisticRegression",
    gauss = "GaussianNB"


#returns list of models 
@app.get("/models")
def greet() -> List[str]: 
    return [model.value for model in Models]

class InputEntry(BaseModel): # 
    value: int

class Output(BaseModel):
    target: int

#takes model name and row. Request body sample: [62, 0, 0, 124, 209, 0, 1, 163, 0, 0, 2, 0, 2]
@app.post("/predict")
def predict(
    model_name: Annotated[Models, Query()], 
    input: Annotated[List[int], Body()]) -> Any:

    return 0


@app.post("/train")
def train_(model_name: Annotated[Models, Query()],
    input: Annotated[List[int], Body()],
    expected_output: Annotated[Output, Body()]) -> Any:
    src.kedro_heart_disease.nodes.check_model = check_model(model_name.value)
    df = pd.DataFrame([input])
    eo = pd.DataFrame({"target": [expected_output.target]})
    print(eo)
    train(src.kedro_heart_disease.nodes.check_model, df, eo)
    return "we did training"


