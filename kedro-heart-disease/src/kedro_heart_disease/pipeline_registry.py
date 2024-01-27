"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
import kedro_heart_disease.pipelines.pycaret_heart_disease as phd
import kedro_heart_disease.pipelines.kedro_heart_disease as khd
import kedro_heart_disease.pipelines.create_models as cm


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    phd_pipeline = phd.create_pipeline()
    khd_pipeline = khd.create_pipeline()
    cm_pipeline = cm.create_pipeline()

    pipelines = find_pipelines()
    #pipelines["__default__"] = sum(pipelines.values())
    return {
        "__default__": cm_pipeline + khd_pipeline,
        "khd": khd_pipeline,
        "cm": cm_pipeline,
        "phd": phd_pipeline,
        "all": sum(pipelines.values())
    }
