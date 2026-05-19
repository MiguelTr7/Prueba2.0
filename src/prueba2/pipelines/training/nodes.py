"""Nodes for the training pipeline."""

from ..model_training import (
    preparar_datos_modelo,
    entrenar_clasificadores,
    entrenar_regresores,
)


def prepare_model_data(df_limpio):
    return preparar_datos_modelo(df_limpio)


def train_classifiers(X_train_clf, y_train_clf, X_test_clf, y_test_clf):
    return entrenar_clasificadores(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf
    )


def train_regressors(X_train_reg, y_train_reg, X_test_reg, y_test_reg):
    return entrenar_regresores(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg
    )
