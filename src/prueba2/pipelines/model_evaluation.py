"""
model_evaluation.py
===================
Funciones de evaluacion, comparacion e interpretacion de modelos de RRHH.

Modulo: src/prueba/model_evaluation.py
Proyecto: Prediccion de desempeno de empleados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


# ------------------------------------------------------------------------------
# EVALUACION DE CLASIFICACION
# ------------------------------------------------------------------------------

def evaluar_clasificacion(
    nombre_modelo: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mostrar_matriz: bool = True
) -> dict:
    """
    Evalua un pipeline de clasificacion y retorna metricas principales.

    Parameters
    ----------
    nombre_modelo : str
        Nombre descriptivo del modelo.
    pipeline : Pipeline
        Pipeline sklearn ya entrenado (fit).
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Etiquetas reales del conjunto de prueba.
    mostrar_matriz : bool, optional
        Si True, grafica la matriz de confusion. Default True.

    Returns
    -------
    dict
        Diccionario con metricas: modelo, accuracy, balanced_accuracy,
        precision, recall, f1_score.
    """
    y_pred = pipeline.predict(X_test)

    resultado = {
        'modelo':            nombre_modelo,
        'accuracy':          accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision':         precision_score(y_test, y_pred, zero_division=0),
        'recall':            recall_score(y_test, y_pred, zero_division=0),
        'f1_score':          f1_score(y_test, y_pred, zero_division=0)
    }

    print(f"\n===== {nombre_modelo} =====")
    print(f"  Accuracy:          {resultado['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {resultado['balanced_accuracy']:.4f}")
    print(f"  Precision:         {resultado['precision']:.4f}")
    print(f"  Recall:            {resultado['recall']:.4f}")
    print(f"  F1-score:          {resultado['f1_score']:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    if mostrar_matriz:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax[0])
        ax[0].set_title(f'Confusion — {nombre_modelo}')

        metricas = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        valores  = [resultado[m] for m in metricas]
        ax[1].barh(metricas, valores, color='steelblue', edgecolor='black', alpha=0.85)
        ax[1].set_xlim(0, 1)
        ax[1].axvline(0.85, color='red', linestyle='--', lw=1.2, label='Meta 85%')
        ax[1].set_title('Metricas')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis='x')
        for i, v in enumerate(valores):
            ax[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    return resultado


# ------------------------------------------------------------------------------
# EVALUACION DE REGRESION
# ------------------------------------------------------------------------------

def evaluar_regresion(
    nombre_modelo: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple:
    """
    Evalua un pipeline de regresion y retorna metricas principales.

    Parameters
    ----------
    nombre_modelo : str
        Nombre descriptivo del modelo.
    pipeline : Pipeline
        Pipeline sklearn ya entrenado (fit).
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Valores reales del conjunto de prueba.

    Returns
    -------
    tuple
        (dict_metricas, array_predicciones)
        dict con: modelo, mae, rmse, r2.
    """
    y_pred = pipeline.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    resultado = {
        'modelo': nombre_modelo,
        'mae':    mean_absolute_error(y_test, y_pred),
        'rmse':   rmse,
        'r2':     r2_score(y_test, y_pred)
    }

    print(f"\n===== {nombre_modelo} =====")
    print(f"  MAE:  {resultado['mae']:.4f}")
    print(f"  RMSE: {resultado['rmse']:.4f}")
    print(f"  R2:   {resultado['r2']:.4f}")

    return resultado, y_pred


# ------------------------------------------------------------------------------
# COMPARACION DE MODELOS
# ------------------------------------------------------------------------------

def comparar_clasificadores(
    resultados: list,
    guardar_plot: str = None
) -> pd.DataFrame:
    """
    Genera tabla y grafico comparativo de modelos de clasificacion.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios retornados por evaluar_clasificacion().
    guardar_plot : str, optional
        Ruta donde guardar el grafico PNG. Si None, no guarda.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa ordenada por F1-score descendente.
    """
    tabla = pd.DataFrame(resultados).sort_values('f1_score', ascending=False)

    plt.figure(figsize=(10, 5))
    colores = ['#1565C0' if f == tabla['f1_score'].max() else '#90CAF9'
               for f in tabla['f1_score']]
    bars = plt.bar(tabla['modelo'], tabla['f1_score'],
                   color=colores, edgecolor='black', alpha=0.9)
    for bar, val in zip(bars, tabla['f1_score']):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.axhline(0.85, color='red', linestyle='--', lw=1.5, label='Meta 85%')
    plt.title('Comparacion F1-score — Clasificacion (desempeno_alto)')
    plt.ylabel('F1-score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    if guardar_plot:
        plt.savefig(guardar_plot, dpi=150, bbox_inches='tight')

    plt.show()
    return tabla


def comparar_regresores(
    resultados: list,
    guardar_plot: str = None
) -> pd.DataFrame:
    """
    Genera tabla y grafico comparativo de modelos de regresion.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios retornados por evaluar_regresion().
    guardar_plot : str, optional
        Ruta donde guardar el grafico PNG. Si None, no guarda.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa ordenada por RMSE ascendente.
    """
    tabla = pd.DataFrame(resultados).sort_values('rmse', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colores_rmse = ['#2E7D32' if m == tabla['modelo'].iloc[0] else '#A5D6A7'
                    for m in tabla['modelo']]
    axes[0].barh(tabla['modelo'], tabla['rmse'],
                 color=colores_rmse, edgecolor='black', alpha=0.9)
    axes[0].set_title('RMSE (menor = mejor)')
    axes[0].set_xlabel('RMSE')
    axes[0].grid(alpha=0.3, axis='x')

    colores_r2 = ['#2E7D32' if r > 0 else '#EF9A9A' for r in tabla['r2']]
    axes[1].barh(tabla['modelo'], tabla['r2'],
                 color=colores_r2, edgecolor='black', alpha=0.9)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5,
                   label='Baseline (predecir media)')
    axes[1].set_title('R2')
    axes[1].set_xlabel('R2')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis='x')

    plt.suptitle('Comparacion — Regresion (puntaje_desempeno)', fontsize=13)
    plt.tight_layout()

    if guardar_plot:
        plt.savefig(guardar_plot, dpi=150, bbox_inches='tight')

    plt.show()
    return tabla


# ------------------------------------------------------------------------------
# VALIDACION CRUZADA
# ------------------------------------------------------------------------------

def validacion_cruzada_clf(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Aplica validacion cruzada estratificada a todos los clasificadores.

    Parameters
    ----------
    pipelines : dict
        Diccionario {nombre: pipeline entrenado}.
    X : pd.DataFrame
        Features completas (train + test).
    y : pd.Series
        Target completo.
    n_splits : int, optional
        Numero de folds. Default 5.

    Returns
    -------
    pd.DataFrame
        Tabla con F1 medio y desviacion estandar por modelo.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = []

    print(f"\n{'='*55}")
    print(f"VALIDACION CRUZADA CLASIFICACION (F1, CV={n_splits})")
    print(f"{'='*55}")

    for nombre, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
        resultados.append({
            'modelo':   nombre,
            'f1_medio': round(scores.mean(), 4),
            'std':      round(scores.std(), 4)
        })
        print(f"  {nombre:<25}  {scores.mean():.4f}  +/-{scores.std():.4f}")

    return pd.DataFrame(resultados).sort_values('f1_medio', ascending=False)


def validacion_cruzada_reg(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Aplica validacion cruzada a todos los regresores.

    Parameters
    ----------
    pipelines : dict
        Diccionario {nombre: pipeline entrenado}.
    X : pd.DataFrame
        Features completas.
    y : pd.Series
        Target continuo.
    n_splits : int, optional
        Numero de folds. Default 5.

    Returns
    -------
    pd.DataFrame
        Tabla con R2 medio y desviacion estandar por modelo.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = []

    print(f"\n{'='*55}")
    print(f"VALIDACION CRUZADA REGRESION (R2, CV={n_splits})")
    print(f"{'='*55}")

    for nombre, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        resultados.append({
            'modelo':   nombre,
            'r2_medio': round(scores.mean(), 4),
            'std':      round(scores.std(), 4)
        })
        print(f"  {nombre:<25}  {scores.mean():.4f}  +/-{scores.std():.4f}")

    return pd.DataFrame(resultados).sort_values('r2_medio', ascending=False)


# ------------------------------------------------------------------------------
# CURVA ROC
# ------------------------------------------------------------------------------

def graficar_roc(
    nombre_modelo: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    guardar_plot: str = None
) -> float:
    """
    Grafica la curva ROC del mejor clasificador y retorna el AUC.

    Parameters
    ----------
    nombre_modelo : str
        Nombre del modelo para el titulo del grafico.
    pipeline : Pipeline
        Pipeline de clasificacion entrenado con probability=True.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiquetas reales.
    guardar_plot : str, optional
        Ruta de guardado del PNG. Si None, no guarda.

    Returns
    -------
    float
        Valor del AUC-ROC.
    """
    modelo = pipeline.named_steps.get('modelo')

    if hasattr(modelo, 'predict_proba'):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(modelo, 'decision_function'):
        y_score = pipeline.decision_function(X_test)
    else:
        print(f"{nombre_modelo} no soporta probabilidades. ROC no disponible.")
        return None

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score   = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='royalblue', lw=2,
             label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--',
             label='Clasificador aleatorio')
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title(f'Curva ROC — {nombre_modelo}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if guardar_plot:
        plt.savefig(guardar_plot, dpi=150, bbox_inches='tight')

    plt.show()
    print(f"AUC-ROC: {auc_score:.4f}")
    return auc_score
