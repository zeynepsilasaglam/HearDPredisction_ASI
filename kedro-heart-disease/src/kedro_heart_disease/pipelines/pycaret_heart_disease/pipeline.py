"""
This is a boilerplate pipeline 'pycaret_heart_disease'
generated using Kedro 0.19.1
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import find_best_model, predict_by_best_model, scoring


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=find_best_model,
                inputs=["heart_disease_data", "params:target_column"],
                outputs=["best_model", "model_setup"],
                name="find_best_model",
            ),
            node(
                func=predict_by_best_model,
                inputs=["best_model", "heart_disease_data", "model_setup"],
                outputs="prediction_best_model",
                name="predict_by_best_model",
            ),
            node(
                func=scoring,
                inputs=["best_model", "prediction_best_model"],
                outputs=None,
                name="scoring",
            ),
        ]
    )
