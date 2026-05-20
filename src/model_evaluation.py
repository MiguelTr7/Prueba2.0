"""
model_evaluation.py
-------------------
Funciones de evaluación y comparación de modelos supervisados para el
proyecto de análisis de desempeño de empleados (SCY1101).

Contenido:
    - evaluar_clasificacion  : métricas completas de un clasificador
    - evaluar_regresion      : métricas completas de un regresor
    - comparar_clasificadores: tabla comparativa de múltiples clasificadores
    - comparar_regresores    : tabla comparativa de múltiples regresores
    - graficar_comparacion_clf: gráfico de barras de métricas de clasificación
    - graficar_comparacion_reg: gráfico de R² y RMSE de regresión
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Evaluación individual
# ---------------------------------------------------------------------------

def evaluar_clasificacion(
    nombre: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mostrar_reporte: bool = True,
) -> pd.Series:
    """Evalúa un clasificador sobre el conjunto de prueba.

    Calcula accuracy, balanced accuracy, precision, recall y F1-score.
    Opcionalmente imprime el classification report y muestra la matriz
    de confusión.

    Parameters
    ----------
    nombre : str
        Etiqueta del modelo para identificarlo en la salida.
    pipeline : sklearn Pipeline
        Pipeline entrenado con el clasificador.
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Target real del conjunto de prueba.
    mostrar_reporte : bool, optional
        Si True, imprime el classification report completo (default: True).

    Returns
    -------
    pd.Series
        Serie con las métricas: modelo, accuracy, balanced_accuracy,
        precision, recall, f1_score.
    """
    y_pred = pipeline.predict(X_test)

    resultado = pd.Series({
        "modelo":            nombre,
        "accuracy":          accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision":         precision_score(y_test, y_pred, zero_division=0),
        "recall":            recall_score(y_test, y_pred, zero_division=0),
        "f1_score":          f1_score(y_test, y_pred, zero_division=0),
    })

    if mostrar_reporte:
        print(f"===== {nombre} =====")
        print(resultado.to_string())
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    return resultado


def evaluar_regresion(
    nombre: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """Evalúa un regresor sobre el conjunto de prueba.

    Parameters
    ----------
    nombre : str
        Etiqueta del modelo para identificarlo en la salida.
    pipeline : sklearn Pipeline
        Pipeline entrenado con el regresor.
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Target real del conjunto de prueba.

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        - Serie con métricas: modelo, mae, rmse, r2.
        - Array de predicciones y_pred.
    """
    y_pred = pipeline.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    resultado = pd.Series({
        "modelo": nombre,
        "mae":    mean_absolute_error(y_test, y_pred),
        "rmse":   rmse,
        "r2":     r2_score(y_test, y_pred),
    })

    print(f"===== {nombre} =====")
    print(resultado.to_string())

    return resultado, y_pred


# ---------------------------------------------------------------------------
# Comparación de múltiples modelos
# ---------------------------------------------------------------------------

def comparar_clasificadores(
    pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evalúa y compara múltiples clasificadores en el mismo test set.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Diccionario nombre → pipeline entrenado.
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Target real del conjunto de prueba.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas ordenada por F1-score descendente.
    """
    filas = []
    for nombre, pipe in pipelines.items():
        filas.append(evaluar_clasificacion(nombre, pipe, X_test, y_test,
                                           mostrar_reporte=False))
    return pd.DataFrame(filas).sort_values("f1_score", ascending=False).reset_index(drop=True)


def comparar_regresores(
    pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evalúa y compara múltiples regresores en el mismo test set.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Diccionario nombre → pipeline entrenado.
    X_test : pd.DataFrame
        Features del conjunto de prueba.
    y_test : pd.Series
        Target real del conjunto de prueba.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas ordenada por R² descendente.
    """
    filas = []
    for nombre, pipe in pipelines.items():
        resultado, _ = evaluar_regresion(nombre, pipe, X_test, y_test)
        filas.append(resultado)
    return pd.DataFrame(filas).sort_values("r2", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def graficar_comparacion_clf(
    df_resultados: pd.DataFrame,
    ruta_plots: str = None,
    nombre_archivo: str = "comparacion_clasificacion.png",
) -> None:
    """Genera gráfico de barras comparando F1, Precision y Recall.

    Parameters
    ----------
    df_resultados : pd.DataFrame
        Tabla generada por comparar_clasificadores().
    ruta_plots : str, optional
        Directorio donde guardar el gráfico. Si es None, solo muestra.
    nombre_archivo : str, optional
        Nombre del archivo PNG (default: 'comparacion_clasificacion.png').
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colores = ["#1565C0" if f == df_resultados["f1_score"].max() else "#90CAF9"
               for f in df_resultados["f1_score"]]
    bars = axes[0].bar(df_resultados["modelo"], df_resultados["f1_score"],
                       color=colores, edgecolor="black", alpha=0.9)
    for bar, val in zip(bars, df_resultados["f1_score"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f"{val:.3f}",
                     ha="center", fontsize=9)
    axes[0].set_title("F1-score por Modelo")
    axes[0].set_ylim(0, df_resultados["f1_score"].max() + 0.15)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.3, axis="y")

    x = range(len(df_resultados))
    axes[1].bar([i - 0.2 for i in x], df_resultados["precision"], width=0.4,
                label="Precision", color="steelblue", alpha=0.85, edgecolor="black")
    axes[1].bar([i + 0.2 for i in x], df_resultados["recall"], width=0.4,
                label="Recall", color="salmon", alpha=0.85, edgecolor="black")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df_resultados["modelo"], rotation=20)
    axes[1].set_title("Precision vs Recall")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Comparación — Modelos de Clasificación", fontsize=13)
    plt.tight_layout()

    if ruta_plots:
        plt.savefig(os.path.join(ruta_plots, nombre_archivo), dpi=150, bbox_inches="tight")
    plt.show()


def graficar_comparacion_reg(
    df_resultados: pd.DataFrame,
    ruta_plots: str = None,
    nombre_archivo: str = "comparacion_regresion.png",
) -> None:
    """Genera gráfico de R² y RMSE para comparar regresores.

    Parameters
    ----------
    df_resultados : pd.DataFrame
        Tabla generada por comparar_regresores().
    ruta_plots : str, optional
        Directorio donde guardar el gráfico. Si es None, solo muestra.
    nombre_archivo : str, optional
        Nombre del archivo PNG (default: 'comparacion_regresion.png').
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colores_r2   = ["#2E7D32" if r == df_resultados["r2"].max()   else "#A5D6A7"
                    for r in df_resultados["r2"]]
    colores_rmse = ["#2E7D32" if r == df_resultados["rmse"].min() else "#A5D6A7"
                    for r in df_resultados["rmse"]]

    axes[0].barh(df_resultados["modelo"], df_resultados["r2"],
                 color=colores_r2, edgecolor="black", alpha=0.9)
    axes[0].axvline(0, color="red", linestyle="--", lw=1.5, label="Baseline")
    axes[0].set_title("R² por Modelo (mayor = mejor)")
    axes[0].set_xlabel("R²")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(df_resultados["modelo"], df_resultados["rmse"],
                 color=colores_rmse, edgecolor="black", alpha=0.9)
    axes[1].set_title("RMSE por Modelo (menor = mejor)")
    axes[1].set_xlabel("RMSE")
    axes[1].grid(alpha=0.3, axis="x")

    plt.suptitle("Comparación — Modelos de Regresión", fontsize=13)
    plt.tight_layout()

    if ruta_plots:
        plt.savefig(os.path.join(ruta_plots, nombre_archivo), dpi=150, bbox_inches="tight")
    plt.show()
