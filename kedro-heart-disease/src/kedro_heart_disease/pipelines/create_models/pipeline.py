"""
This is a boilerplate pipeline 'create_models'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import create_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=create_models,
                inputs="heart_disease_data",
                outputs=["rand_for_model", "knn_model", "gauss_model", "current_model"],
                name="models",
            ),
    ])
