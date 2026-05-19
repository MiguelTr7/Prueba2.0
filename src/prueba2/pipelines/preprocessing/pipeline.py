from kedro.pipeline import Pipeline, node

from .nodes import run_preprocessing


def create_pipeline() -> Pipeline:
    """Create the preprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=run_preprocessing,
                inputs=None,
                outputs="df_limpio",
                name="run_preprocessing",
            ),
        ]
    )
