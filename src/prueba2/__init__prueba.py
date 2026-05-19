"""
prueba
======
Paquete de Machine Learning para predicción de desempeño de empleados.

Módulos disponibles
-------------------
- data_preprocessing  : limpieza, transformación y preparación de datos
- model_training      : construcción, pipelines y entrenamiento de modelos
- model_evaluation    : evaluación, métricas y validación cruzada
- hyperparameter_tuning : optimización con GridSearchCV y Optuna

Uso rápido
----------
>>> import sys
>>> sys.path.insert(0, r'C:\\Users\\Arturo\\Desktop\\Prueba1\\src')
>>>
>>> from prueba.data_preprocessing import limpiar_datos_basico
>>> from prueba.model_training import crear_preprocesador, obtener_clasificadores
>>> from prueba.model_evaluation import evaluar_clasificacion, validacion_cruzada_clf
>>> from prueba.hyperparameter_tuning import optimizar_optuna_clf
"""

__version__ = '1.0.0'
__author__  = 'Arturo'
__project__ = 'Predicción de Desempeño de Empleados — SCY1101'

# Importaciones principales expuestas a nivel de paquete.
from prueba.data_preprocessing import (
    cargar_datos_crudos,
    normalizar_categorias,
    agregar_ausencias,
    agregar_capacitaciones,
    construir_dataset_evaluaciones,
    crear_features_empleado,
    limpiar_datos_basico
)

from prueba.model_training import (
    crear_preprocesador,
    crear_pipeline,
    obtener_clasificadores,
    obtener_regresores,
    obtener_variables_modelo
)

from prueba.model_evaluation import (
    evaluar_clasificacion,
    evaluar_regresion,
    comparar_clasificadores,
    comparar_regresores,
    validacion_cruzada_clf,
    validacion_cruzada_reg,
    graficar_roc
)

from prueba.hyperparameter_tuning import (
    optimizar_gridsearch_clf,
    optimizar_gridsearch_reg,
    optimizar_optuna_clf,
    optimizar_optuna_reg,
    ajustar_umbral_optimo,
    comparar_enfoques_tuning
)

__all__ = [
    # data_preprocessing
    'cargar_datos_crudos',
    'normalizar_categorias',
    'agregar_ausencias',
    'agregar_capacitaciones',
    'construir_dataset_evaluaciones',
    'crear_features_empleado',
    'limpiar_datos_basico',
    # model_training
    'crear_preprocesador',
    'crear_pipeline',
    'obtener_clasificadores',
    'obtener_regresores',
    'obtener_variables_modelo',
    # model_evaluation
    'evaluar_clasificacion',
    'evaluar_regresion',
    'comparar_clasificadores',
    'comparar_regresores',
    'validacion_cruzada_clf',
    'validacion_cruzada_reg',
    'graficar_roc',
    # hyperparameter_tuning
    'optimizar_gridsearch_clf',
    'optimizar_gridsearch_reg',
    'optimizar_optuna_clf',
    'optimizar_optuna_reg',
    'ajustar_umbral_optimo',
    'comparar_enfoques_tuning',
]
