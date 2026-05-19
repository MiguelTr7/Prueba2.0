from kedro.pipeline import Pipeline, node

from .nodes import (
    evaluate_classifier_models,
    evaluate_regressor_models,
)


def create_pipeline() -> Pipeline:
    """Create the evaluation pipeline."""
    return Pipeline(
        [
            node(
                func=evaluate_classifier_models,
                inputs=["pipelines_clf", "X_test_clf", "y_test_clf"],
                outputs="resultados_clf",
                name="evaluate_classifier_models",
            ),
            node(
                func=evaluate_regressor_models,
                inputs=["pipelines_reg", "X_test_reg", "y_test_reg"],
                outputs="resultados_reg",
                name="evaluate_regressor_models",
            ),
        ]
    )
