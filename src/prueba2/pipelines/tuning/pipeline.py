from kedro.pipeline import Pipeline, node

from .nodes import (
    tune_classifier_models,
    tune_regressor_models,
)


def create_pipeline() -> Pipeline:
    """Create the hyperparameter tuning pipeline."""
    return Pipeline(
        [
            node(
                func=tune_classifier_models,
                inputs=["X_train_clf", "y_train_clf", "X_test_clf", "y_test_clf"],
                outputs="tuning_clf",
                name="tune_classifier_models",
            ),
            node(
                func=tune_regressor_models,
                inputs=["X_train_reg", "y_train_reg", "X_test_reg", "y_test_reg"],
                outputs="tuning_reg",
                name="tune_regressor_models",
            ),
        ]
    )
