"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_predictions_all_models, report_accuracy, split_data, models, predict



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["heart_disease_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=models,
                inputs=["X_train", "y_train"],
                outputs=["random_forest", "knn", "logistic_regression", "gaussian"],
                name="models",
            ),
            node(
                func=make_predictions_all_models,
                inputs=["random_forest", "knn", "logistic_regression", "gaussian", "X_test"],
                outputs=["random_forest_predict", "knn_predict", "logistic_regression_predict", "gaussian_predict"],
                name="make_predictions_all_models",
            ),
            node(
                func=report_accuracy,
                inputs=["random_forest_predict", "knn_predict", "logistic_regression_predict", "gaussian_predict", "y_test"],
                outputs=None,
                name="report_accuracy",
            ),
            node(
                func=predict,
                inputs=["random_forest", "parameters"],
                outputs=None,
                name="predict",
            ),
        ]
    )
