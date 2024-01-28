"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train, predict, split_data, model_score


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
                func=train,
                inputs=["rf_model", "X_train", "y_train"],
                outputs="trained_model",
                name="train",
            ),
            node(
                func=predict,
                inputs=["trained_model", "X_test"],
                outputs="prediction",
                name="predict",
            ),
            node(
                func=model_score,
                inputs="trained_model",
                outputs=None,
                name="model_score",
            ),
        ]
    )
