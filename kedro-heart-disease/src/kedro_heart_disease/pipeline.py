"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train, predict, split_data, models, model_score


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
                outputs=None,
                name="train",
            ),
            node(
                func=predict,
                inputs=["current_model", "X_test"],
                outputs="prediction",
                name="predict",
            ),
            node(
                func=model_score,
                inputs="current_model",
                outputs=None,
                name="model_score",
            ),
        ]
    )
