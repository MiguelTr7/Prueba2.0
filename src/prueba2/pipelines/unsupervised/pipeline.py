from kedro.pipeline import Pipeline, node

from .nodes import (
    prepare_unsupervised_data,
    run_unsupervised_analysis,
)


def create_pipeline() -> Pipeline:
    """Create the unsupervised analysis pipeline."""
    return Pipeline(
        [
            node(
                func=prepare_unsupervised_data,
                inputs="df_limpio",
                outputs="X_unsupervised",
                name="prepare_unsupervised_data",
            ),
            node(
                func=run_unsupervised_analysis,
                inputs="X_unsupervised",
                outputs="unsupervised_summary",
                name="run_unsupervised_analysis",
            ),
        ]
    )
