"""Nodes for the hyperparameter tuning pipeline."""

from os import path

from ..hyperparameter_tuning import (
    optimizar_gridsearch_clf,
    optimizar_gridsearch_reg,
    optimizar_optuna_clf,
    optimizar_optuna_reg,
    optimizar_randomizedsearch_clf,
    optimizar_randomizedsearch_reg,
    exportar_resultados_tuning,
)
from ..model_training import crear_preprocesador


def tune_classifier_models(X_train_clf, y_train_clf, X_test_clf, y_test_clf):
    preprocessor = crear_preprocesador(X_train_clf)
    grid = optimizar_gridsearch_clf(
        X_train_clf,
        y_train_clf,
        preprocessor,
        cv=3,
    )
    randomized = optimizar_randomizedsearch_clf(
        X_train_clf,
        y_train_clf,
        preprocessor,
        cv=3,
        n_iter=15,
    )
    study, _ = optimizar_optuna_clf(
        X_train_clf,
        y_train_clf,
        X_test_clf,
        y_test_clf,
        preprocessor,
        n_trials=10,
    )

    resultados = {
        'grid': {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
        },
        'randomized': {
            'best_params': randomized.best_params_,
            'best_score': randomized.best_score_,
        },
        'optuna': {
            'best_params': study.best_params if study else None,
            'best_value': study.best_value if study else None,
        },
    }
    exportar_resultados_tuning(
        resultados,
        path.join('data', '08_reporting', 'tuning_classifier_summary.json'),
    )
    return resultados


def tune_regressor_models(X_train_reg, y_train_reg, X_test_reg, y_test_reg):
    preprocessor = crear_preprocesador(X_train_reg)
    grid = optimizar_gridsearch_reg(
        X_train_reg,
        y_train_reg,
        preprocessor,
        cv=3,
    )
    randomized = optimizar_randomizedsearch_reg(
        X_train_reg,
        y_train_reg,
        preprocessor,
        cv=3,
        n_iter=15,
    )
    study, _ = optimizar_optuna_reg(
        X_train_reg,
        y_train_reg,
        X_test_reg,
        y_test_reg,
        preprocessor,
        n_trials=10,
    )

    resultados = {
        'grid': {
            'best_params': grid.best_params_,
            'best_score': abs(grid.best_score_),
        },
        'randomized': {
            'best_params': randomized.best_params_,
            'best_score': abs(randomized.best_score_),
        },
        'optuna': {
            'best_params': study.best_params if study else None,
            'best_value': study.best_value if study else None,
        },
    }
    exportar_resultados_tuning(
        resultados,
        path.join('data', '08_reporting', 'tuning_regressor_summary.json'),
    )
    return resultados
