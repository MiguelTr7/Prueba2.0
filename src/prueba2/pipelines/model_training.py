"""
model_training.py
=================
Definicion, construccion y entrenamiento de modelos supervisados de RRHH.

Modulo: src/prueba/model_training.py
Proyecto: Prediccion de desempeno de empleados
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


# ------------------------------------------------------------------------------
# PREPROCESADOR
# ------------------------------------------------------------------------------

def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Crea un preprocesador ColumnTransformer para el dataset de RRHH.

    Aplica:
    - Variables numericas: imputacion con mediana + StandardScaler.
    - Variables categoricas: imputacion con moda + OneHotEncoder.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de features del cual se detectan tipos de columnas.

    Returns
    -------
    ColumnTransformer
        Preprocesador listo para incluir en un Pipeline de sklearn.
    """
    vars_num = X.select_dtypes(include=np.number).columns.tolist()
    vars_cat = X.select_dtypes(exclude=np.number).columns.tolist()

    transformador_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    transformador_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocesador = ColumnTransformer([
        ('num', transformador_num, vars_num),
        ('cat', transformador_cat, vars_cat)
    ])

    return preprocesador


def crear_pipeline(preprocesador: ColumnTransformer, modelo) -> Pipeline:
    """
    Construye un Pipeline con preprocesador y modelo.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador generado por crear_preprocesador().
    modelo : estimator sklearn
        Modelo clasificador o regresor de sklearn.

    Returns
    -------
    Pipeline
        Pipeline listo para .fit() y .predict().
    """
    return Pipeline([
        ('preprocesamiento', preprocesador),
        ('modelo', modelo)
    ])


# ------------------------------------------------------------------------------
# CATALOGOS DE MODELOS
# ------------------------------------------------------------------------------

def obtener_clasificadores() -> dict:
    """
    Retorna el catalogo de clasificadores utilizados en el proyecto.

    Todos sin class_weight porque el target desempeno_alto usa P50
    (balance 50/50 entre clases).

    Returns
    -------
    dict
        Diccionario {nombre: instancia_modelo} de clasificadores.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=8
        ),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(
            random_state=42,
            probability=True
        ),
        'Gaussian NB': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1
        )
    }


def obtener_regresores() -> dict:
    """
    Retorna el catalogo de regresores utilizados en el proyecto.

    Returns
    -------
    dict
        Diccionario {nombre: instancia_modelo} de regresores.
    """
    return {
        'Linear Regression':         LinearRegression(),
        'Decision Tree Regressor':   DecisionTreeRegressor(
            random_state=42,
            max_depth=6
        ),
        'Random Forest Regressor':   RandomForestRegressor(
            random_state=42,
            n_estimators=200,
            max_depth=8
        ),
        'KNN Regressor':             KNeighborsRegressor(),
        'SVR':                       SVR()
    }


def preparar_datos_modelo(df: pd.DataFrame):
    """Prepara los datos de entrenamiento y evaluación para los modelos."""
    X = df[obtener_variables_modelo(df)]
    y_clf = df['desempeno_alto']
    y_reg = df['puntaje_desempeno']

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X,
        y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    return (
        X_train_clf,
        X_test_clf,
        y_train_clf,
        y_test_clf,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    )


# ------------------------------------------------------------------------------
# ENTRENAMIENTO
# ------------------------------------------------------------------------------

def entrenar_clasificadores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    modelos: dict = None
) -> tuple:
    """
    Entrena todos los clasificadores del catalogo y retorna pipelines entrenados.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (desempeno_alto).
    X_test : pd.DataFrame
        Features de evaluacion.
    y_test : pd.Series
        Target de evaluacion.
    modelos : dict, optional
        Diccionario de modelos. Si es None, usa obtener_clasificadores().

    Returns
    -------
    tuple
        (pipelines_entrenados: dict, pipelines_dict: dict)
        donde pipelines_entrenados es {nombre: pipeline fitted}.
    """
    if modelos is None:
        modelos = obtener_clasificadores()

    pipelines = {}
    for nombre, modelo in modelos.items():
        preprocesador = crear_preprocesador(X_train)
        pipeline = crear_pipeline(preprocesador, modelo)
        pipeline.fit(X_train, y_train)
        pipelines[nombre] = pipeline
        print(f"  Entrenado: {nombre}")

    return pipelines


def entrenar_regresores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    modelos: dict = None
) -> dict:
    """
    Entrena todos los regresores del catalogo y retorna pipelines entrenados.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (puntaje_desempeno).
    X_test : pd.DataFrame
        Features de evaluacion.
    y_test : pd.Series
        Target de evaluacion.
    modelos : dict, optional
        Diccionario de modelos. Si es None, usa obtener_regresores().

    Returns
    -------
    dict
        Diccionario {nombre: pipeline fitted}.
    """
    if modelos is None:
        modelos = obtener_regresores()

    pipelines = {}
    for nombre, modelo in modelos.items():
        preprocesador = crear_preprocesador(X_train)
        pipeline = crear_pipeline(preprocesador, modelo)
        pipeline.fit(X_train, y_train)
        pipelines[nombre] = pipeline
        print(f"  Entrenado: {nombre}")

    return pipelines


# ------------------------------------------------------------------------------
# VARIABLES PREDICTORAS
# ------------------------------------------------------------------------------

def obtener_variables_modelo(df: pd.DataFrame) -> list:
    """
    Retorna la lista de variables predictoras del proyecto.

    Excluye targets, identificadores y variables con data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo con todas las columnas.

    Returns
    -------
    list
        Lista de nombres de columnas a usar como features.
    """
    excluir = [
        'id_empleado', 'periodo',
        'puntaje_desempeno', 'desempeno_alto',
        'desempeno_bajo', 'riesgo_rotacion',
        'puntaje_desempeno_original', 'evaluador',
        'id_evaluacion'
    ]

    variables_base = [
        'competencias_tecnicas', 'competencias_blandas', 'periodo_num',
        'total_dias_ausencia', 'num_ausencias', 'promedio_dias_ausencia',
        'max_dias_ausencia', 'ausencia_alta',
        'num_capacitaciones', 'total_horas_capacitacion',
        'promedio_nota_capacitacion', 'max_nota_capacitacion',
        'sin_capacitacion', 'capacitacion_intensiva',
        'competencia_combinada', 'eficiencia_capacitacion',
        'brecha_nota_capacitacion', 'riesgo_operacional_rrhh',
        'departamento', 'cargo', 'tipo_contrato', 'jornada',
        'total_dias_ausencia_log', 'promedio_dias_ausencia_log',
        'max_dias_ausencia_log', 'num_capacitaciones_log',
        'total_horas_capacitacion_log', 'riesgo_operacional_rrhh_log',
        'competencia_combinada_log'
    ]

    return [v for v in variables_base if v in df.columns and v not in excluir]
