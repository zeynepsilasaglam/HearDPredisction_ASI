"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_current_model, train, predict, split_data, accuracy


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="heart_disease_data",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=get_current_model,
                inputs=None,
                outputs="current_model",
                name="get_current_model",
            ),
            node(
                func=train,
                inputs=["current_model", "heart_disease_data"],
                outputs="trained_model",
                name="train",
            ),
            node(
                func=predict,
                inputs=["trained_model", "parameters"],
                outputs="prediction",
                name="predict",
            ),
             node(
                func=accuracy,
                inputs=["trained_model", "X_test", "y_test"],
                outputs=None,
                name="accuracy",
            ),
        ]
    )
