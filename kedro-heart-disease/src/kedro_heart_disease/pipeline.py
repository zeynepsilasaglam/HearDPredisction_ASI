"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_current_model, train, predict, split_data, accuracy, models


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
                func=models,
                inputs="heart_disease_data",
                outputs=["rand_for_model", "knn_model", "log_reg_model", "gauss_model", "current_model"],
                name="models",
            ),
            node(
                func=train,
                inputs=["current_model", "X_train", "y_train"],
                outputs="score",
                name="train",
            ),
            node(
                func=predict,
                inputs=["trained_model", "X_test"],
                outputs="prediction",
                name="predict",
            ),
        ]
    )
