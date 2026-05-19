"""
hyperparameter_tuning.py
========================
Funciones para optimizacion de hiperparametros de modelos de RRHH.

Incluye GridSearchCV, Optuna y ajuste de umbral de probabilidad.

Modulo: src/prueba/hyperparameter_tuning.py
Proyecto: Prediccion de desempeno de empleados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score, mean_squared_error, classification_report
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_DISPONIBLE = True
except ImportError:
    OPTUNA_DISPONIBLE = False


# ------------------------------------------------------------------------------
# GRIDSEARCHCV
# ------------------------------------------------------------------------------

def optimizar_gridsearch_clf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocesador,
    cv: int = 3
) -> GridSearchCV:
    """
    Aplica GridSearchCV sobre RandomForestClassifier para clasificacion.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (desempeno_alto).
    preprocesador : ColumnTransformer
        Preprocesador de crear_preprocesador().
    cv : int, optional
        Numero de folds para validacion cruzada. Default 3.

    Returns
    -------
    GridSearchCV
        Objeto ajustado con el mejor estimador en .best_estimator_.
    """
    pipeline = Pipeline([
        ('preprocesamiento', preprocesador),
        ('modelo', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'modelo__n_estimators':     [100, 200],
        'modelo__max_depth':        [6, 8, 10],
        'modelo__min_samples_split': [2, 5],
        'modelo__min_samples_leaf':  [1, 2]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("GridSearchCV Clasificacion — Mejores parametros:")
    print(grid.best_params_)
    print(f"Mejor F1 CV: {grid.best_score_:.4f}")

    return grid


def optimizar_gridsearch_reg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocesador,
    cv: int = 3
) -> GridSearchCV:
    """
    Aplica GridSearchCV sobre RandomForestRegressor para regresion.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (puntaje_desempeno).
    preprocesador : ColumnTransformer
        Preprocesador de crear_preprocesador().
    cv : int, optional
        Numero de folds. Default 3.

    Returns
    -------
    GridSearchCV
        Objeto ajustado con el mejor estimador en .best_estimator_.
    """
    pipeline = Pipeline([
        ('preprocesamiento', preprocesador),
        ('modelo', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'modelo__n_estimators':     [100, 200],
        'modelo__max_depth':        [6, 8, 10],
        'modelo__min_samples_split': [2, 5],
        'modelo__min_samples_leaf':  [1, 2]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("GridSearchCV Regresion — Mejores parametros:")
    print(grid.best_params_)
    print(f"Mejor RMSE CV: {abs(grid.best_score_):.4f}")

    return grid


# ------------------------------------------------------------------------------
# OPTUNA
# ------------------------------------------------------------------------------

def optimizar_optuna_clf(
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
    preprocesador,
    n_trials: int = 20
):
    """
    Aplica busqueda de hiperparametros con Optuna para clasificacion.

    Maximiza el F1-score sobre el conjunto de prueba.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    X_test : pd.DataFrame
        Features de evaluacion.
    y_test : pd.Series
        Target de evaluacion.
    preprocesador : ColumnTransformer
        Preprocesador de crear_preprocesador().
    n_trials : int, optional
        Numero de trials de Optuna. Default 20.

    Returns
    -------
    tuple
        (study, pipeline_optuna) si Optuna esta disponible.
        None si Optuna no esta instalado.
    """
    if not OPTUNA_DISPONIBLE:
        print("Optuna no disponible. Instala con: pip install optuna")
        return None, None

    def objetivo(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 300),
            'max_depth':         trial.suggest_int('max_depth', 4, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5)
        }
        pipe = Pipeline([
            ('preprocesamiento', preprocesador),
            ('modelo', RandomForestClassifier(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        return f1_score(y_test, pipe.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(objetivo, n_trials=n_trials)

    print(f"Optuna Clasificacion — Mejores parametros:")
    print(study.best_params)
    print(f"Mejor F1: {study.best_value:.4f}")

    # Reentrenar con los mejores parametros.
    pipeline_optuna = Pipeline([
        ('preprocesamiento', preprocesador),
        ('modelo', RandomForestClassifier(**study.best_params, random_state=42))
    ])
    pipeline_optuna.fit(X_train, y_train)

    return study, pipeline_optuna


def optimizar_optuna_reg(
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
    preprocesador,
    n_trials: int = 20
):
    """
    Aplica busqueda de hiperparametros con Optuna para regresion.

    Minimiza el RMSE sobre el conjunto de prueba.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    X_test : pd.DataFrame
        Features de evaluacion.
    y_test : pd.Series
        Target de evaluacion.
    preprocesador : ColumnTransformer
        Preprocesador de crear_preprocesador().
    n_trials : int, optional
        Numero de trials de Optuna. Default 20.

    Returns
    -------
    tuple
        (study, pipeline_optuna) si Optuna esta disponible.
        None si Optuna no instalado.
    """
    if not OPTUNA_DISPONIBLE:
        print("Optuna no disponible. Instala con: pip install optuna")
        return None, None

    def objetivo(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 400),
            'max_depth':         trial.suggest_int('max_depth', 4, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5)
        }
        pipe = Pipeline([
            ('preprocesamiento', preprocesador),
            ('modelo', RandomForestRegressor(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))

    study = optuna.create_study(direction='minimize')
    study.optimize(objetivo, n_trials=n_trials)

    print(f"Optuna Regresion — Mejores parametros:")
    print(study.best_params)
    print(f"Mejor RMSE: {study.best_value:.4f}")

    pipeline_optuna = Pipeline([
        ('preprocesamiento', preprocesador),
        ('modelo', RandomForestRegressor(**study.best_params, random_state=42))
    ])
    pipeline_optuna.fit(X_train, y_train)

    return study, pipeline_optuna


# ------------------------------------------------------------------------------
# AJUSTE DE UMBRAL
# ------------------------------------------------------------------------------

def ajustar_umbral_optimo(
    pipeline,
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
    nombre_modelo: str = 'Modelo',
    guardar_plot:  str = None
) -> tuple:
    """
    Encuentra el umbral de probabilidad que maximiza el F1-score.

    Util para modelos de clasificacion que retornan probabilidades
    (predict_proba), como Logistic Regression y Random Forest.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline de clasificacion entrenado con probability=True.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiquetas reales.
    nombre_modelo : str, optional
        Nombre para el titulo del grafico.
    guardar_plot : str, optional
        Ruta de guardado del grafico PNG.

    Returns
    -------
    tuple
        (umbral_optimo: float, f1_optimo: float, y_pred_optimo: array)
    """
    if not hasattr(pipeline.named_steps.get('modelo'), 'predict_proba'):
        print(f"{nombre_modelo} no soporta predict_proba.")
        return None, None, None

    y_proba   = pipeline.predict_proba(X_test)[:, 1]
    umbrales  = np.arange(0.20, 0.80, 0.01)
    f1_scores = [
        f1_score(y_test, (y_proba >= u).astype(int), zero_division=0)
        for u in umbrales
    ]

    umbral_optimo = umbrales[np.argmax(f1_scores)]
    f1_optimo     = max(f1_scores)
    y_pred_optimo = (y_proba >= umbral_optimo).astype(int)

    print(f"Umbral por defecto (0.50): F1 = {f1_score(y_test, pipeline.predict(X_test), zero_division=0):.4f}")
    print(f"Umbral optimo ({umbral_optimo:.2f}):    F1 = {f1_optimo:.4f}")
    print(classification_report(y_test, y_pred_optimo, zero_division=0))

    plt.figure(figsize=(9, 4))
    plt.plot(umbrales, f1_scores, color='steelblue', lw=2)
    plt.axvline(umbral_optimo, color='red', linestyle='--',
                label=f'Umbral optimo = {umbral_optimo:.2f}  F1 = {f1_optimo:.3f}')
    plt.axvline(0.50, color='gray', linestyle=':', label='Umbral default = 0.50')
    plt.axhline(0.85, color='green', linestyle='--', label='Meta 85%')
    plt.title(f'F1-score por Umbral — {nombre_modelo}')
    plt.xlabel('Umbral')
    plt.ylabel('F1-score')
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if guardar_plot:
        plt.savefig(guardar_plot, dpi=150, bbox_inches='tight')

    plt.show()

    return umbral_optimo, f1_optimo, y_pred_optimo


# ------------------------------------------------------------------------------
# COMPARACION FINAL
# ------------------------------------------------------------------------------

def comparar_enfoques_tuning(
    resultados_base:  dict,
    resultados_grid:  dict,
    resultados_optuna: dict,
    tipo: str = 'clf'
) -> pd.DataFrame:
    """
    Genera tabla comparativa entre modelo base, GridSearch y Optuna.

    Parameters
    ----------
    resultados_base : dict
        Metricas del modelo base.
    resultados_grid : dict
        Metricas del modelo GridSearchCV.
    resultados_optuna : dict
        Metricas del modelo Optuna.
    tipo : str, optional
        'clf' para clasificacion, 'reg' para regresion. Default 'clf'.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa de los tres enfoques.
    """
    tabla = pd.DataFrame([resultados_base, resultados_grid, resultados_optuna])

    metrica_clave = 'f1_score' if tipo == 'clf' else 'rmse'
    ascendente    = False if tipo == 'clf' else True

    print(f"\n{'='*55}")
    print(f"COMPARACION: Base vs GridSearchCV vs Optuna")
    print(f"{'='*55}")
    print(tabla.sort_values(metrica_clave, ascending=ascendente).to_string(index=False))

    return tabla
