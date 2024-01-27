"""
This is a boilerplate pipeline 'create_models'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import optimize_, fit_

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=optimize_,
                inputs="heart_disease_data",
                outputs=["rf_model", "knn_model", "gnb_model"],
                name="optimize",
            ),
            node(
                func=fit_,
                inputs="heart_disease_data",
                outputs=None,
                name="fit-persist",
            ),   
        ]
    )

