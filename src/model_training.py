"""
model_training.py
-----------------
Definición y entrenamiento de modelos supervisados para el proyecto de
análisis de desempeño de empleados (SCY1101).

Contenido:
    - obtener_modelos_clasificacion : diccionario de clasificadores configurados
    - obtener_modelos_regresion     : diccionario de regresores configurados
    - entrenar_modelo               : entrena un pipeline y retorna el pipeline ajustado
    - entrenar_todos                : itera sobre un dict de modelos y retorna resultados
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from data_preprocessing import crear_preprocesador


# ---------------------------------------------------------------------------
# Catálogos de modelos
# ---------------------------------------------------------------------------

def obtener_modelos_clasificacion() -> dict:
    """Retorna un diccionario con los clasificadores del proyecto.

    Todos los modelos tienen `class_weight='balanced'` donde aplique para
    compensar el desbalance de clases (~73% clase 0 / ~27% clase 1).
    Se fija `random_state=42` para reproducibilidad.

    Returns
    -------
    dict[str, estimator]
        Clave: nombre descriptivo. Valor: instancia de clasificador sklearn.
    """
    return {
        "Decision Tree":     DecisionTreeClassifier(
                                 max_depth=5, class_weight="balanced", random_state=42),
        "Logistic Reg.":     LogisticRegression(
                                 class_weight="balanced", max_iter=1000, random_state=42),
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "SVM":               SVC(
                                 kernel="rbf", class_weight="balanced",
                                 probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
                                 n_estimators=200, max_depth=4, random_state=42),
        "Random Forest":     RandomForestClassifier(
                                 n_estimators=200, max_depth=8,
                                 class_weight="balanced", random_state=42),
    }


def obtener_modelos_regresion() -> dict:
    """Retorna un diccionario con los regresores del proyecto.

    Returns
    -------
    dict[str, estimator]
        Clave: nombre descriptivo. Valor: instancia de regresor sklearn.
    """
    return {
        "Linear Reg.":       LinearRegression(),
        "KNN Regressor":     KNeighborsRegressor(n_neighbors=7),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
        "SVR":               SVR(kernel="rbf", C=1.0),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=200, max_depth=4, random_state=42),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200, max_depth=8, random_state=42),
    }


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar_modelo(
    nombre: str,
    modelo,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    paso_modelo: str = "clf",
) -> Pipeline:
    """Construye un Pipeline (preprocesador + modelo) y lo entrena.

    Parameters
    ----------
    nombre : str
        Nombre descriptivo del modelo (solo para logging).
    modelo : sklearn estimator
        Instancia del modelo a entrenar.
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    paso_modelo : str, optional
        Nombre del paso del modelo dentro del Pipeline. Usar 'clf' para
        clasificadores y 'reg' para regresores (default: 'clf').

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline ajustado listo para predecir.
    """
    pipe = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        (paso_modelo, modelo),
    ])
    pipe.fit(X_train, y_train)
    print(f"  Entrenado: {nombre}")
    return pipe


def entrenar_todos(
    modelos: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    paso_modelo: str = "clf",
) -> dict:
    """Entrena todos los modelos del diccionario y retorna los pipelines ajustados.

    Parameters
    ----------
    modelos : dict[str, estimator]
        Diccionario nombre → instancia de modelo (ver obtener_modelos_*).
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    paso_modelo : str, optional
        Nombre del paso del modelo en el Pipeline (default: 'clf').

    Returns
    -------
    dict[str, Pipeline]
        Diccionario nombre → pipeline entrenado.
    """
    pipelines = {}
    for nombre, modelo in modelos.items():
        pipelines[nombre] = entrenar_modelo(nombre, modelo, X_train, y_train, paso_modelo)
    return pipelines
