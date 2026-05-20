"""
hyperparameter_tuning.py
------------------------
Funciones para la optimización de hiperparámetros de modelos de ML para el
proyecto de análisis de desempeño de empleados (SCY1101).

Cubre tres estrategias:
    1. GridSearchCV       — búsqueda exhaustiva en grilla definida manualmente.
    2. RandomizedSearchCV — muestreo aleatorio desde distribuciones estadísticas.
    3. Optuna             — optimización bayesiana con validación cruzada sin
                           data leakage (CV sobre train únicamente).

Contenido:
    - tune_grid_search_clf      : GridSearchCV para clasificación
    - tune_grid_search_reg      : GridSearchCV para regresión
    - tune_random_search_clf    : RandomizedSearchCV para clasificación
    - tune_random_search_reg    : RandomizedSearchCV para regresión
    - tune_optuna_clf           : Optuna para clasificación (RF)
    - tune_optuna_reg           : Optuna para regresión (RF)
    - comparar_resultados_tuning: tabla comparativa de los tres enfoques
"""

import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_score, StratifiedKFold, KFold,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data_preprocessing import crear_preprocesador


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------

def tune_grid_search_clf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "f1",
) -> GridSearchCV:
    """Optimiza RandomForestClassifier con GridSearchCV.

    Recorre una grilla fija de hiperparámetros y selecciona la combinación
    con mayor F1-score en validación cruzada sobre el conjunto de entrenamiento.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (binario).
    cv : int, optional
        Número de folds de validación cruzada (default: 5).
    scoring : str, optional
        Métrica de optimización (default: 'f1').

    Returns
    -------
    GridSearchCV
        Objeto ajustado; usar `.best_estimator_` para el pipeline final.
    """
    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "modelo__n_estimators":      [100, 200, 300],
        "modelo__max_depth":         [6, 8, 10],
        "modelo__min_samples_split": [2, 5],
        "modelo__min_samples_leaf":  [1, 2],
        "modelo__class_weight":      ["balanced", None],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"[GridSearchCV clf] Mejores params: {search.best_params_}")
    print(f"[GridSearchCV clf] Mejor {scoring} CV: {search.best_score_:.4f}")
    return search


def tune_grid_search_reg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> GridSearchCV:
    """Optimiza RandomForestRegressor con GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (continuo).
    cv : int, optional
        Número de folds (default: 5).
    scoring : str, optional
        Métrica de optimización (default: 'neg_root_mean_squared_error').

    Returns
    -------
    GridSearchCV
        Objeto ajustado; usar `.best_estimator_` para el pipeline final.
    """
    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestRegressor(random_state=42)),
    ])

    param_grid = {
        "modelo__n_estimators":      [100, 200, 300],
        "modelo__max_depth":         [6, 8, 10],
        "modelo__min_samples_split": [2, 5],
        "modelo__min_samples_leaf":  [1, 2],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"[GridSearchCV reg] Mejores params: {search.best_params_}")
    print(f"[GridSearchCV reg] Mejor RMSE CV: {abs(search.best_score_):.4f}")
    return search


# ---------------------------------------------------------------------------
# RandomizedSearchCV
# ---------------------------------------------------------------------------

def tune_random_search_clf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Optimiza RandomForestClassifier con RandomizedSearchCV.

    Muestrea `n_iter` combinaciones aleatorias de distribuciones estadísticas,
    lo que cubre un espacio de búsqueda más amplio que GridSearchCV con el
    mismo costo computacional.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (binario).
    n_iter : int, optional
        Número de combinaciones a muestrear (default: 30).
    cv : int, optional
        Número de folds de validación cruzada (default: 5).
    scoring : str, optional
        Métrica de optimización (default: 'f1').
    random_state : int, optional
        Semilla para reproducibilidad (default: 42).

    Returns
    -------
    RandomizedSearchCV
        Objeto ajustado; usar `.best_estimator_` para el pipeline final.
    """
    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestClassifier(random_state=random_state)),
    ])

    param_distributions = {
        "modelo__n_estimators":      randint(100, 400),
        "modelo__max_depth":         randint(4, 15),
        "modelo__min_samples_split": randint(2, 10),
        "modelo__min_samples_leaf":  randint(1, 5),
        "modelo__class_weight":      ["balanced", None],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"[RandomizedSearchCV clf] Mejores params: {search.best_params_}")
    print(f"[RandomizedSearchCV clf] Mejor {scoring} CV: {search.best_score_:.4f}")
    return search


def tune_random_search_reg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Optimiza RandomForestRegressor con RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (continuo).
    n_iter : int, optional
        Número de combinaciones a muestrear (default: 30).
    cv : int, optional
        Número de folds (default: 5).
    scoring : str, optional
        Métrica de optimización (default: 'neg_root_mean_squared_error').
    random_state : int, optional
        Semilla para reproducibilidad (default: 42).

    Returns
    -------
    RandomizedSearchCV
        Objeto ajustado; usar `.best_estimator_` para el pipeline final.
    """
    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestRegressor(random_state=random_state)),
    ])

    param_distributions = {
        "modelo__n_estimators":      randint(100, 400),
        "modelo__max_depth":         randint(4, 15),
        "modelo__min_samples_split": randint(2, 10),
        "modelo__min_samples_leaf":  randint(1, 5),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"[RandomizedSearchCV reg] Mejores params: {search.best_params_}")
    print(f"[RandomizedSearchCV reg] Mejor RMSE CV: {abs(search.best_score_):.4f}")
    return search


# ---------------------------------------------------------------------------
# Optuna (requiere: pip install optuna)
# ---------------------------------------------------------------------------

def tune_optuna_clf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    cv: int = 5,
):
    """Optimiza RandomForestClassifier con Optuna (búsqueda bayesiana).

    IMPORTANTE: la evaluación se realiza con cross_val_score sobre X_train
    únicamente — el test set no participa en ningún trial, eliminando el
    data leakage que ocurre cuando se evalúa sobre X_test dentro del objetivo.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (binario).
    n_trials : int, optional
        Número de trials de Optuna (default: 30).
    cv : int, optional
        Número de folds de validación cruzada (default: 5).

    Returns
    -------
    tuple[optuna.study.Study, Pipeline]
        - study: objeto Optuna con el historial de trials.
        - pipeline: Pipeline ajustado con los mejores hiperparámetros.

    Raises
    ------
    ImportError
        Si Optuna no está instalado.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna no instalado. Ejecuta: pip install optuna") from exc

    cv_opt = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def objetivo(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      100, 400),
            "max_depth":         trial.suggest_int("max_depth",          4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split",  2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",   1, 5),
            "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
        }
        pipe = Pipeline([
            ("preprocesamiento", crear_preprocesador(X_train)),
            ("modelo", RandomForestClassifier(**params, random_state=42)),
        ])
        return cross_val_score(pipe, X_train, y_train,
                               cv=cv_opt, scoring="f1", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objetivo, n_trials=n_trials)

    print(f"[Optuna clf] Mejor F1 CV: {study.best_value:.4f}")
    print(f"[Optuna clf] Mejores params: {study.best_params}")

    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestClassifier(**study.best_params, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    return study, pipeline


def tune_optuna_reg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    cv: int = 5,
):
    """Optimiza RandomForestRegressor con Optuna (búsqueda bayesiana).

    CV se realiza exclusivamente sobre X_train — sin data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (continuo).
    n_trials : int, optional
        Número de trials (default: 30).
    cv : int, optional
        Número de folds (default: 5).

    Returns
    -------
    tuple[optuna.study.Study, Pipeline]
        - study: objeto Optuna.
        - pipeline: Pipeline ajustado.

    Raises
    ------
    ImportError
        Si Optuna no está instalado.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna no instalado. Ejecuta: pip install optuna") from exc

    cv_opt = KFold(n_splits=cv, shuffle=True, random_state=42)

    def objetivo(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      100, 400),
            "max_depth":         trial.suggest_int("max_depth",          4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split",  2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",   1, 5),
        }
        pipe = Pipeline([
            ("preprocesamiento", crear_preprocesador(X_train)),
            ("modelo", RandomForestRegressor(**params, random_state=42)),
        ])
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv_opt,
                                 scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objetivo, n_trials=n_trials)

    print(f"[Optuna reg] Mejor RMSE CV: {study.best_value:.4f}")
    print(f"[Optuna reg] Mejores params: {study.best_params}")

    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(X_train)),
        ("modelo", RandomForestRegressor(**study.best_params, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    return study, pipeline


# ---------------------------------------------------------------------------
# Comparación de enfoques
# ---------------------------------------------------------------------------

def comparar_resultados_tuning(
    resultados: list,
    metrica_principal: str = "f1_score",
) -> pd.DataFrame:
    """Construye una tabla comparativa de los enfoques de optimización.

    Parameters
    ----------
    resultados : list of pd.Series
        Lista de Series de métricas (salida de evaluar_clasificacion o
        evaluar_regresion).
    metrica_principal : str, optional
        Columna por la que ordenar la tabla (default: 'f1_score').

    Returns
    -------
    pd.DataFrame
        Tabla comparativa ordenada por la métrica principal.
    """
    tabla = pd.DataFrame(resultados)
    if metrica_principal in tabla.columns:
        ascendente = metrica_principal in ("rmse", "mae")
        tabla = tabla.sort_values(metrica_principal, ascending=ascendente)
    return tabla.reset_index(drop=True)
