from kedro.pipeline import Pipeline, node

from .nodes import (
    prepare_model_data,
    train_classifiers,
    train_regressors,
)


def create_pipeline() -> Pipeline:
    """Create the training pipeline."""
    return Pipeline(
        [
            node(
                func=prepare_model_data,
                inputs="df_limpio",
                outputs=[
                    "X_train_clf",
                    "X_test_clf",
                    "y_train_clf",
                    "y_test_clf",
                    "X_train_reg",
                    "X_test_reg",
                    "y_train_reg",
                    "y_test_reg",
                ],
                name="prepare_model_data",
            ),
            node(
                func=train_classifiers,
                inputs=["X_train_clf", "y_train_clf", "X_test_clf", "y_test_clf"],
                outputs="pipelines_clf",
                name="train_classifiers",
            ),
            node(
                func=train_regressors,
                inputs=["X_train_reg", "y_train_reg", "X_test_reg", "y_test_reg"],
                outputs="pipelines_reg",
                name="train_regressors",
            ),
        ]
    )
