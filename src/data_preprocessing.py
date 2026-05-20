"""
data_preprocessing.py
---------------------
Funciones de limpieza, transformación y preparación de datos para el
proyecto de análisis de desempeño de empleados (SCY1101).

Contenido:
    - cargar_dataset          : carga dataset_rrhh_limpio.csv
    - crear_features_derivadas: lag, sesgo evaluador e interacciones
    - definir_targets         : targets de clasificación y regresión
    - crear_preprocesador     : ColumnTransformer con imputer + scaler/OHE
    - obtener_variables_modelo: lista filtrada de variables predictoras
"""

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_dataset(project_root: str) -> pd.DataFrame:
    """Carga el dataset limpio de RRHH desde la ruta estándar del proyecto.

    Parameters
    ----------
    project_root : str
        Ruta raíz del proyecto (p. ej. r'C:/Users/Arturo/prueba2').

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos listos para feature engineering.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta esperada.
    """
    ruta = os.path.join(project_root, "data", "05_model_input", "dataset_rrhh_limpio.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Dataset no encontrado: {ruta}")
    df = pd.read_csv(ruta)
    print(f"Dataset cargado: {df.shape[0]} registros × {df.shape[1]} variables")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables derivadas de alto valor predictivo al DataFrame.

    Variables creadas:
        - prev_puntaje      : puntaje del período anterior del mismo empleado
                              (lag-1); correlación ~0.47 con el target.
        - evaluador_media   : promedio histórico que asigna cada evaluador;
                              captura sesgo sistemático (std_entre_eval=1.82
                              vs std_intra=0.84 → ~60% varianza del evaluador).
        - dept_media        : promedio por departamento; captura diferencias
                              estructurales entre áreas.
        - comp_asistencia   : competencia_combinada × (1 − ausencia_alta).
        - nota_cap_ponderada: promedio_nota × log(1 + n_capacitaciones).
        - ratio_tec_blanda  : competencias_tecnicas / competencias_blandas.
        - cap_score         : max_nota × (1 − sin_capacitacion).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame base con las columnas originales del dataset limpio.

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas agregadas (in-place copy).
    """
    df = df.copy()

    # Lag feature: puntaje del período anterior del mismo empleado.
    df_orden = df.sort_values(["id_empleado", "periodo_num"])
    df["prev_puntaje"] = (
        df_orden.groupby("id_empleado")["puntaje_desempeno"].shift(1).values
    )

    # Sesgo del evaluador y del departamento.
    ev_media = df.groupby("evaluador")["puntaje_desempeno"].mean()
    df["evaluador_media"] = df["evaluador"].map(ev_media)

    dept_media = df.groupby("departamento")["puntaje_desempeno"].mean()
    df["dept_media"] = df["departamento"].map(dept_media)

    # Variables de interacción.
    df["comp_asistencia"]    = df["competencia_combinada"] * (1 - df["ausencia_alta"])
    df["nota_cap_ponderada"] = (
        df["promedio_nota_capacitacion"] * np.log1p(df["num_capacitaciones"])
    )
    df["ratio_tec_blanda"] = (
        df["competencias_tecnicas"] / df["competencias_blandas"].clip(lower=0.1)
    )
    df["cap_score"] = df["max_nota_capacitacion"] * (1 - df["sin_capacitacion"])

    return df


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def definir_targets(df: pd.DataFrame, percentil: float = 0.75) -> pd.DataFrame:
    """Define la columna de clasificación binaria `desempeno_alto`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna `puntaje_desempeno`.
    percentil : float, optional
        Umbral percentil para definir "alto desempeño" (default: 0.75 → P75).

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna `desempeno_alto` agregada.
    """
    df = df.copy()
    umbral = df["puntaje_desempeno"].quantile(percentil)
    df["desempeno_alto"] = (df["puntaje_desempeno"] >= umbral).astype(int)
    print(f"Umbral desempeno_alto (P{int(percentil*100)}): {umbral:.2f}")
    print(f"  Clase 0: {(df['desempeno_alto']==0).sum()} | Clase 1: {(df['desempeno_alto']==1).sum()}")
    return df


# ---------------------------------------------------------------------------
# Variables predictoras
# ---------------------------------------------------------------------------

VARIABLES_BASE = [
    "total_dias_ausencia", "promedio_dias_ausencia", "max_dias_ausencia", "ausencia_alta",
    "num_capacitaciones", "total_horas_capacitacion", "promedio_nota_capacitacion",
    "max_nota_capacitacion", "sin_capacitacion", "capacitacion_intensiva",
    "competencias_tecnicas", "competencias_blandas", "competencia_combinada",
    "eficiencia_capacitacion", "brecha_nota_capacitacion", "riesgo_operacional_rrhh",
    "departamento", "cargo", "tipo_contrato", "jornada", "evaluador",
    "periodo_num", "prev_puntaje", "evaluador_media", "dept_media",
    "total_dias_ausencia_log", "promedio_dias_ausencia_log", "max_dias_ausencia_log",
    "num_capacitaciones_log", "total_horas_capacitacion_log",
    "riesgo_operacional_rrhh_log", "competencia_combinada_log",
    "comp_asistencia", "nota_cap_ponderada", "ratio_tec_blanda", "cap_score",
]


def obtener_variables_modelo(df: pd.DataFrame) -> list:
    """Retorna la lista de variables predictoras presentes en el DataFrame.

    Filtra `VARIABLES_BASE` conservando solo las columnas que existen en `df`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se usará para entrenamiento.

    Returns
    -------
    list of str
        Variables predictoras disponibles.
    """
    disponibles = [col for col in VARIABLES_BASE if col in df.columns]
    print(f"Variables modelo: {len(disponibles)} / {len(VARIABLES_BASE)}")
    return disponibles


# ---------------------------------------------------------------------------
# Preprocesador
# ---------------------------------------------------------------------------

def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocesador de sklearn para el DataFrame X dado.

    Aplica por tipo de columna:
        - Numéricas : SimpleImputer(median) → StandardScaler
        - Categóricas: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')

    Parameters
    ----------
    X : pd.DataFrame
        Features de entrenamiento. Se usa solo para detectar tipos de columna;
        el preprocesador se ajusta luego con .fit().

    Returns
    -------
    ColumnTransformer
        Preprocesador listo para encadenar en un sklearn Pipeline.
    """
    vars_num = X.select_dtypes(include=np.number).columns.tolist()
    vars_cat = X.select_dtypes(exclude=np.number).columns.tolist()

    transformador_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    transformador_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", transformador_num, vars_num),
        ("cat", transformador_cat, vars_cat),
    ])
