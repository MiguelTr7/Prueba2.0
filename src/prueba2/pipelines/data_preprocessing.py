"""
data_preprocessing.py
=====================
Funciones de limpieza, transformacion y preparacion del dataset de RRHH.

Modulo: src/prueba/data_preprocessing.py
Proyecto: Prediccion de desempeno de empleados
"""

import pandas as pd
import numpy as np
import os


# ------------------------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------------------------

def cargar_datos_crudos(ruta_raw: str) -> dict:
    """
    Carga los cuatro archivos CSV fuente del proyecto de RRHH.

    Parameters
    ----------
    ruta_raw : str
        Ruta a la carpeta que contiene los archivos CSV.

    Returns
    -------
    dict
        Diccionario con DataFrames: 'empleados', 'ausencias',
        'capacitaciones', 'evaluaciones'.

    Raises
    ------
    FileNotFoundError
        Si alguno de los archivos no existe en la ruta indicada.
    """
    archivos = {
        'empleados':      'empleados.csv',
        'ausencias':      'ausencias.csv',
        'capacitaciones': 'capacitaciones.csv',
        'evaluaciones':   'evaluaciones.csv'
    }

    datos = {}
    for nombre, archivo in archivos.items():
        ruta_completa = os.path.join(ruta_raw, archivo)
        if not os.path.exists(ruta_completa):
            raise FileNotFoundError(
                f"Archivo no encontrado: {ruta_completa}"
            )
        datos[nombre] = pd.read_csv(ruta_completa)
        print(f"  {nombre}: {datos[nombre].shape}")

    return datos


# ------------------------------------------------------------------------------
# NORMALIZACION
# ------------------------------------------------------------------------------

def normalizar_categorias(datos: dict) -> dict:
    """
    Normaliza columnas categoricas con inconsistencias de capitalizacion.

    Aplica .strip().title() a columnas de texto en cada tabla.

    Parameters
    ----------
    datos : dict
        Diccionario de DataFrames retornado por cargar_datos_crudos().

    Returns
    -------
    dict
        Mismo diccionario con columnas normalizadas.
    """
    for col in ['departamento', 'cargo', 'tipo_contrato', 'jornada']:
        if col in datos['empleados'].columns:
            datos['empleados'][col] = (
                datos['empleados'][col].astype(str).str.strip().str.title()
            )

    if 'tipo_ausencia' in datos['ausencias'].columns:
        datos['ausencias']['tipo_ausencia'] = (
            datos['ausencias']['tipo_ausencia'].astype(str).str.strip().str.title()
        )

    if 'estado' in datos['capacitaciones'].columns:
        datos['capacitaciones']['estado'] = (
            datos['capacitaciones']['estado'].astype(str).str.strip().str.title()
        )

    return datos


# ------------------------------------------------------------------------------
# AGREGACION
# ------------------------------------------------------------------------------

def agregar_ausencias(ausencias: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega registros de ausencias a nivel de empleado.

    Parameters
    ----------
    ausencias : pd.DataFrame
        Tabla de ausencias con columnas: id_empleado, dias, id_ausencia.

    Returns
    -------
    pd.DataFrame
        Una fila por empleado con metricas agregadas de ausencias.
    """
    aus = ausencias.dropna(subset=['id_empleado', 'dias']).copy()
    aus['id_empleado'] = aus['id_empleado'].astype(int)
    aus['dias'] = pd.to_numeric(aus['dias'], errors='coerce')
    aus = aus[aus['dias'] >= 0]

    aus_agg = aus.groupby('id_empleado').agg(
        total_dias_ausencia    = ('dias', 'sum'),
        num_ausencias          = ('id_ausencia', 'count'),
        promedio_dias_ausencia = ('dias', 'mean'),
        max_dias_ausencia      = ('dias', 'max')
    ).reset_index()

    for col in ['total_dias_ausencia', 'promedio_dias_ausencia', 'max_dias_ausencia']:
        p1  = aus_agg[col].quantile(0.01)
        p99 = aus_agg[col].quantile(0.99)
        aus_agg[col] = aus_agg[col].clip(p1, p99)

    return aus_agg


def agregar_capacitaciones(capacitaciones: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega registros de capacitaciones a nivel de empleado.

    Parameters
    ----------
    capacitaciones : pd.DataFrame
        Tabla de capacitaciones con columnas: id_empleado, horas,
        nota_final, id_capacitacion.

    Returns
    -------
    pd.DataFrame
        Una fila por empleado con metricas agregadas de capacitaciones.
    """
    cap = capacitaciones.dropna(subset=['id_empleado']).copy()
    cap['id_empleado'] = cap['id_empleado'].astype(int)
    cap['horas']       = pd.to_numeric(cap['horas'],      errors='coerce')
    cap['nota_final']  = pd.to_numeric(cap['nota_final'], errors='coerce')

    cap_agg = cap.groupby('id_empleado').agg(
        num_capacitaciones         = ('id_capacitacion', 'count'),
        total_horas_capacitacion   = ('horas',            'sum'),
        promedio_nota_capacitacion = ('nota_final',        'mean'),
        max_nota_capacitacion      = ('nota_final',        'max')
    ).reset_index()

    for col in ['total_horas_capacitacion', 'num_capacitaciones']:
        p99 = cap_agg[col].quantile(0.99)
        cap_agg[col] = cap_agg[col].clip(0, p99)

    return cap_agg


# ------------------------------------------------------------------------------
# CONSTRUCCION DEL DATASET
# ------------------------------------------------------------------------------

def construir_dataset_evaluaciones(
    evaluaciones: pd.DataFrame,
    empleados:    pd.DataFrame,
    aus_agg:      pd.DataFrame,
    cap_agg:      pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el dataset a nivel de evaluacion (una fila por registro).

    Usa evaluaciones.csv como tabla base y une datos estaticos del empleado,
    ausencias agregadas y capacitaciones agregadas. Esta decision permite
    aprovechar competencias_tecnicas y competencias_blandas del periodo exacto
    como predictores directos del puntaje de desempeno.

    Parameters
    ----------
    evaluaciones : pd.DataFrame
        Tabla base de evaluaciones por periodo.
    empleados : pd.DataFrame
        Datos estaticos del empleado.
    aus_agg : pd.DataFrame
        Ausencias agregadas (retornado por agregar_ausencias()).
    cap_agg : pd.DataFrame
        Capacitaciones agregadas (retornado por agregar_capacitaciones()).

    Returns
    -------
    pd.DataFrame
        Dataset integrado con aproximadamente 478 filas.
    """
    ev = evaluaciones.dropna(subset=['id_empleado', 'puntaje_desempeno']).copy()
    ev['id_empleado'] = ev['id_empleado'].astype(int)

    # Clipping P1-P99 en el target continuo.
    p1  = ev['puntaje_desempeno'].quantile(0.01)
    p99 = ev['puntaje_desempeno'].quantile(0.99)
    ev['puntaje_desempeno'] = ev['puntaje_desempeno'].clip(p1, p99)

    # Imputar nulos en competencias con la mediana.
    for col in ['competencias_tecnicas', 'competencias_blandas']:
        if col in ev.columns:
            ev[col] = ev[col].fillna(ev[col].median())

    # Codificar periodo como entero ordinal.
    periodos = {'2022-S1': 1, '2022-S2': 2, '2023-S1': 3,
                '2023-S2': 4, '2024-S1': 5}
    ev['periodo_num'] = ev['periodo'].map(periodos).fillna(0).astype(int)

    # Unir datos del empleado.
    emp = empleados.dropna(subset=['id_empleado']).copy()
    emp['id_empleado'] = emp['id_empleado'].astype(int)

    df = ev.merge(
        emp[['id_empleado', 'departamento', 'cargo', 'tipo_contrato', 'jornada']],
        on='id_empleado', how='left'
    )
    df = df.merge(aus_agg, on='id_empleado', how='left')
    df = df.merge(cap_agg, on='id_empleado', how='left')

    cols_aus = ['total_dias_ausencia', 'num_ausencias',
                'promedio_dias_ausencia', 'max_dias_ausencia']
    cols_cap = ['num_capacitaciones', 'total_horas_capacitacion',
                'promedio_nota_capacitacion', 'max_nota_capacitacion']
    df[cols_aus] = df[cols_aus].fillna(0)
    df[cols_cap] = df[cols_cap].fillna(0)

    return df


# ------------------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------------------

def crear_features_empleado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas de RRHH para el dataset a nivel de evaluacion.

    Genera flags de riesgo, variables combinadas, transformaciones
    logaritmicas y el target de clasificacion (desempeno_alto).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset integrado (de construir_dataset_evaluaciones()).

    Returns
    -------
    pd.DataFrame
        Dataset enriquecido con variables derivadas y targets listos
        para modelado.
    """
    df = df.copy()

    # Flags de ausencias.
    umbral_aus = df['total_dias_ausencia'].quantile(0.75)
    df['ausencia_alta'] = (df['total_dias_ausencia'] >= umbral_aus).astype(int)

    # Flags de capacitaciones.
    df['sin_capacitacion'] = (df['num_capacitaciones'] == 0).astype(int)
    umbral_int = df['total_horas_capacitacion'].quantile(0.75)
    df['capacitacion_intensiva'] = (
        df['total_horas_capacitacion'] >= umbral_int
    ).astype(int)

    # Variables de competencias (especificas del periodo).
    df['competencia_combinada'] = (
        df['competencias_tecnicas'] + df['competencias_blandas']
    )
    df['eficiencia_capacitacion'] = np.where(
        df['total_horas_capacitacion'] > 0,
        df['promedio_nota_capacitacion'] / df['total_horas_capacitacion'],
        0
    )
    df['brecha_nota_capacitacion'] = (
        df['max_nota_capacitacion'] - df['promedio_nota_capacitacion']
    )

    # Variables de riesgo.
    umbral_bajo = df['puntaje_desempeno'].quantile(0.25)
    df['desempeno_bajo'] = (df['puntaje_desempeno'] <= umbral_bajo).astype(int)
    df['riesgo_rotacion'] = (
        (df['ausencia_alta'] == 1) & (df['desempeno_bajo'] == 1)
    ).astype(int)
    df['riesgo_operacional_rrhh'] = (
        df['ausencia_alta'] + df['sin_capacitacion'] + df['desempeno_bajo']
    )

    # Target clasificacion (P50 = balance 50/50).
    umbral_alto = df['puntaje_desempeno'].quantile(0.50)
    df['desempeno_alto'] = (df['puntaje_desempeno'] >= umbral_alto).astype(int)

    # Transformaciones logaritmicas.
    cols_log = [
        'total_dias_ausencia', 'promedio_dias_ausencia',
        'max_dias_ausencia', 'num_capacitaciones',
        'total_horas_capacitacion', 'riesgo_operacional_rrhh',
        'competencia_combinada'
    ]
    for col in cols_log:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    return df


# ------------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ------------------------------------------------------------------------------

def limpiar_datos_basico(ruta_raw: str, ruta_output: str) -> pd.DataFrame:
    """
    Pipeline completo de preparacion del dataset de RRHH.

    Ejecuta en orden: carga -> normalizacion -> agregacion ->
    construccion -> feature engineering -> guardado.

    Parameters
    ----------
    ruta_raw : str
        Ruta a la carpeta con los CSV fuente.
    ruta_output : str
        Ruta donde se guardara dataset_rrhh_limpio.csv.

    Returns
    -------
    pd.DataFrame
        Dataset final listo para modelado (~478 filas).
    """
    print("=" * 50)
    print("PIPELINE DE PREPROCESAMIENTO RRHH")
    print("=" * 50)

    datos   = cargar_datos_crudos(ruta_raw)
    datos   = normalizar_categorias(datos)
    aus_agg = agregar_ausencias(datos['ausencias'])
    cap_agg = agregar_capacitaciones(datos['capacitaciones'])
    df      = construir_dataset_evaluaciones(
        datos['evaluaciones'], datos['empleados'], aus_agg, cap_agg
    )
    df = crear_features_empleado(df)

    os.makedirs(ruta_output, exist_ok=True)
    ruta_guardar = os.path.join(ruta_output, 'dataset_rrhh_limpio.csv')
    df.to_csv(ruta_guardar, index=False)

    print(f"\nDataset guardado en: {ruta_guardar}")
    print(f"Shape final: {df.shape}")
    print(f"Target clf — Clase 0: {(df['desempeno_alto']==0).sum()} | "
          f"Clase 1: {(df['desempeno_alto']==1).sum()}")
    print("=" * 50)

    return df
