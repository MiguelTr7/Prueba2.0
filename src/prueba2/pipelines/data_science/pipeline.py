"""
Data Science pipeline wrapper.

This package exposes a Kedro pipeline that composes the separated
preprocessing, training, evaluation and tuning pipelines.
"""

from kedro.pipeline import Pipeline

from ..evaluation import create_pipeline as create_evaluation_pipeline
from ..preprocessing import create_pipeline as create_preprocessing_pipeline
from ..training import create_pipeline as create_training_pipeline
from ..tuning import create_pipeline as create_tuning_pipeline
from ..unsupervised import create_pipeline as create_unsupervised_pipeline


def create_pipeline() -> Pipeline:
    """Create the full data science workflow pipeline."""
    return (
        create_preprocessing_pipeline()
        + create_training_pipeline()
        + create_tuning_pipeline()
        + create_evaluation_pipeline()
        + create_unsupervised_pipeline()
    )
