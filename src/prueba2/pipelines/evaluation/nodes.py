"""Nodes for the evaluation pipeline."""

from ..model_evaluation import (
    evaluar_clasificacion,
    evaluar_regresion,
)


def evaluate_classifier_models(pipelines_clf, X_test_clf, y_test_clf):
    resultados = []
    for nombre, pipeline in pipelines_clf.items():
        resultados.append(
            evaluar_clasificacion(
                nombre,
                pipeline,
                X_test_clf,
                y_test_clf,
                mostrar_matriz=False,
            )
        )
    return resultados


def evaluate_regressor_models(pipelines_reg, X_test_reg, y_test_reg):
    resultados = []
    for nombre, pipeline in pipelines_reg.items():
        resultado, _ = evaluar_regresion(
            nombre,
            pipeline,
            X_test_reg,
            y_test_reg,
        )
        resultados.append(resultado)
    return resultados
